from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import importlib.util
import numpy as np

from ..base import CyberBackend


@dataclass
class CybORGAdapterConfig:
    # Number of CybORG simulation steps for the EnterpriseScenarioGenerator
    steps: int = 60

    # Name of a blue agent to target for action-label resolution / observation selection.
    # NOTE: CC4 is multi-agent; rewards often arrive as a dict across blue agents.
    agent_name: str = "blue_agent_0"

    # CybORG controller mode: "sim" (default) or "emu" (if you have an emulation backend)
    environment: str = "sim"


class CybORGAdapter(CyberBackend):
    """
    Adapter to use CybORG (CAGE Challenge 4 Enterprise scenario) as the cyber backend.

    This adapter is designed to be:
    - Optional (CybORG is an external dependency)
    - Reproducible (we only require lightweight runtime deps for the simulator)
    - Robust to CC4 packaging quirks:
        * CC4's Wrappers package imports RLlib wrappers (ray) eagerly via __init__.py.
          We avoid that by importing BlueFixedActionWrapper directly from its file.

    Output contract (to V-CHIMERA):
      - We provide a small stable summary observation (detection_conf, severity, evidence_available)
        plus raw_obs for debugging/inspection.
      - We provide metrics including cyber_harm = - team_reward (higher => worse).
    """

    def __init__(self, cfg: CybORGAdapterConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(0)
        self._cyborg = None
        self._wrapped = None

        # Internal proxy state (we do not depend on CybORG internals for these)
        self.detection_conf = 0.15
        self.evidence_progress = 0.0
        self.t = 0

        # Resolved at reset
        self._idx: Dict[str, int] = {}
        self._primary_agent: str = cfg.agent_name

    def _require(self) -> None:
        try:
            import CybORG  # noqa: F401
        except Exception as e:
            raise ImportError(
                "CybORG is not installed/importable. For CC4, clone cage-challenge-4 and add it to PYTHONPATH (or write a .pth file) "
                "then ensure minimal runtime deps (gym, pygame, etc.) are installed."
            ) from e

    def _import_blue_fixed_action_wrapper(self):
        """Import BlueFixedActionWrapper without triggering CybORG.Agents.Wrappers.__init__ (which imports ray)."""
        import CybORG as cyborg_pkg  # package

        cyborg_root = Path(cyborg_pkg.__file__).resolve().parent
        bf_path = cyborg_root / "Agents" / "Wrappers" / "BlueFixedActionWrapper.py"
        if not bf_path.exists():
            raise ImportError(f"Could not locate BlueFixedActionWrapper.py at expected path: {bf_path}")

        spec = importlib.util.spec_from_file_location("_cc4_BlueFixedActionWrapper", bf_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create import spec for {bf_path}")

        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        if not hasattr(mod, "BlueFixedActionWrapper"):
            raise ImportError("BlueFixedActionWrapper symbol not found after loading module.")
        return mod.BlueFixedActionWrapper

    def reset(self, seed: int) -> Dict[str, Any]:
        self._require()
        from CybORG import CybORG
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.detection_conf = float(self.rng.uniform(0.10, 0.20))
        self.evidence_progress = 0.0

        sg = EnterpriseScenarioGenerator(steps=int(self.cfg.steps))
        self._cyborg = CybORG(sg, self.cfg.environment)

        BlueFixedActionWrapper = self._import_blue_fixed_action_wrapper()
        self._wrapped = BlueFixedActionWrapper(self._cyborg)

        r = self._wrapped.reset()
        if isinstance(r, tuple) and len(r) == 2:
            obs, info = r
        else:
            obs, info = r, {}

        # Determine a primary agent key that actually exists in observations (CC4 is multi-agent)
        if isinstance(obs, dict) and len(obs) > 0:
            if self.cfg.agent_name in obs:
                self._primary_agent = self.cfg.agent_name
            else:
                self._primary_agent = next(iter(obs.keys()))
        else:
            self._primary_agent = self.cfg.agent_name

        # Build mapping from our abstract action names -> CybORG action indices for the primary agent
        labels = self._wrapped.action_labels(self._primary_agent)

        def find_idx(substrs):
            for s in substrs:
                for i, lab in enumerate(labels):
                    if s.lower() in str(lab).lower():
                        return int(i)
            return 0  # fallback to Sleep

        self._idx = {
            "noop": find_idx(["sleep"]),
            "monitor": find_idx(["monitor"]),
            "hunt": find_idx(["remove", "analyse", "analyze"]),
            "patch": find_idx(["deploydecoy", "deploy decoy", "deploy"]),
            "isolate": find_idx(["blocktraffic", "block traffic", "block"]),
            "restore": find_idx(["restore"]),
        }

        return self._summarize(obs, reward_team=0.0, reward_agent=0.0)

    @staticmethod
    def _to_float_map(d: Any) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not isinstance(d, dict):
            return out
        for k, v in d.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out

    def _extract_rewards(self, rewards: Any) -> Tuple[float, float, Dict[str, float]]:
        """Return (team_reward, primary_agent_reward, reward_map)."""
        if isinstance(rewards, dict):
            rmap = self._to_float_map(rewards)
            # Team reward: prefer __all__ if present, else sum of agent rewards
            if "__all__" in rmap:
                r_team = float(rmap["__all__"])
            else:
                r_team = float(sum(v for k, v in rmap.items() if k != "__all__"))
            # Primary agent reward: best-effort
            r_agent = float(rmap.get(self._primary_agent, 0.0))
            if r_agent == 0.0 and self._primary_agent not in rmap:
                # If only one non-__all__ key exists, use it
                keys = [k for k in rmap.keys() if k != "__all__"]
                if len(keys) == 1:
                    r_agent = float(rmap[keys[0]])
            return r_team, r_agent, rmap

        # Scalar reward fallback
        try:
            r = float(rewards)
        except Exception:
            r = 0.0
        return r, r, {"__scalar__": r}

    def _summarize(self, obs: Any, reward_team: float, reward_agent: float) -> Dict[str, Any]:
        # Severity proxy: map (more negative team reward) -> higher severity
        # reward_team == 0 => 0.5; negative => closer to 1.0
        sev = float(1.0 / (1.0 + np.exp(0.15 * reward_team)))
        evidence_available = bool(self.evidence_progress >= 0.55 and self.detection_conf >= 0.55)

        return {
            "t": self.t,
            "detection_conf": float(np.clip(self.detection_conf, 0.0, 1.0)),
            "severity": float(np.clip(sev, 0.0, 1.0)),
            # proxies (unknown without TrueStateWrapper)
            "compromised_frac": float("nan"),
            "services_down": float("nan"),
            "exfil_risk": float("nan"),
            "ransomware": float("nan"),
            "evidence_available": evidence_available,
            # expose rewards for debugging / analysis
            "reward_team": float(reward_team),
            "reward_primary": float(reward_agent),
            "primary_agent": str(self._primary_agent),
            "raw_obs": obs,
        }

    def step(self, action: str, modifiers: Dict[str, float]) -> Tuple[Dict[str, Any], Dict[str, float], bool]:
        self._require()
        if self._wrapped is None:
            raise RuntimeError("Call reset() before step().")

        a = action.lower().strip()
        idx = self._idx.get(a, self._idx.get("noop", 0))

        # CC4 wrapper expects dict[agent_name -> action_index]. We'll act only on the primary agent;
        # other agents will default to their wrapper-defined behavior.
        obs, rewards, terminated, truncated, info = self._wrapped.step({self._primary_agent: idx})

        r_team, r_agent, rmap = self._extract_rewards(rewards)

        # update internal detection/evidence proxies based on defender intent
        if a in ("monitor",):
            self.detection_conf = float(
                min(1.0, self.detection_conf + 0.07 * (0.7 + 0.6 * modifiers.get("reporting", 0.6)))
            )
            self.evidence_progress = float(min(1.0, self.evidence_progress + 0.06))
        elif a in ("hunt",):
            self.detection_conf = float(min(1.0, self.detection_conf + 0.03))
            self.evidence_progress = float(min(1.0, self.evidence_progress + 0.10))
        elif a in ("restore", "patch", "isolate"):
            self.detection_conf = float(max(0.0, self.detection_conf - 0.01))
        else:
            self.detection_conf = float(max(0.0, self.detection_conf - 0.02))

        self.t += 1

        # Termination (CC4 returns dicts)
        done = False
        if isinstance(terminated, dict):
            done = bool(terminated.get("__all__", any(bool(v) for v in terminated.values())))
        if isinstance(truncated, dict):
            done = bool(done or truncated.get("__all__", any(bool(v) for v in truncated.values())))

        # Select per-agent observation for summarization if possible
        if isinstance(obs, dict) and len(obs) > 0:
            obs_primary = obs.get(self._primary_agent, next(iter(obs.values())))
        else:
            obs_primary = obs

        summary_obs = self._summarize(obs_primary, reward_team=r_team, reward_agent=r_agent)

        metrics = {
            "cyber_harm": float(-r_team),  # harm proxy: higher => worse
            "cyborg_reward_team": float(r_team),
            "cyborg_reward_primary": float(r_agent),
            "detection_conf": float(summary_obs["detection_conf"]),
            "severity": float(summary_obs["severity"]),
            "compromised_frac": float("nan"),
            "services_down": float("nan"),
            "exfil_risk": float("nan"),
            "ransomware": float("nan"),
            "evidence_available": float(1.0 if summary_obs["evidence_available"] else 0.0),
        }

        return summary_obs, metrics, done
