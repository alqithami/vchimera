from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .ledger import LedgerWriter
from .rules import NormRule, NoPunishmentBeamRule
from ..envs.ssd_env import extract_obs


@dataclass
class IMLConfig:
    enabled: bool = False

    # Detection properties (noise model)
    p_detect_true: float = 0.9   # P(detect | violation)
    p_detect_false: float = 0.01 # P(detect | no violation)

    # Sanction magnitude (added as negative reward)
    sanction: float = 0.5

    # Due process / review model:
    # If a sanction was applied, with probability p_review we review.
    # If the sanction was a false positive and review occurs, it is overturned (refund).
    p_review: float = 0.0

    # Logging
    write_ledger: bool = False
    ledger_path: Optional[str] = None

    # Rule configuration
    rules: List[NormRule] = field(default_factory=lambda: [NoPunishmentBeamRule()])


class IMLWrapper:
    """Environment wrapper implementing a basic institution:

    1) Computes ground-truth rule violations from (obs, action, reward, info).
    2) Applies noisy detection (TP/FP rates).
    3) Applies sanctions (reward shaping) on detected violations.
    4) Optionally performs a review/appeal that can overturn false positives.
    5) Emits structured info and optional audit ledger.

    This wrapper assumes an RLlib-style multi-agent env:
        obs_dict = env.reset()
        next_obs, rewards, dones, infos = env.step(action_dict)

    It does **not** require RLlib itself.
    """

    def __init__(self, env: Any, cfg: IMLConfig, run_dir: Optional[Path] = None, seed: Optional[int] = None):
        self.env = env
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._episode = 0

        self._prev_obs: Optional[Dict[str, np.ndarray]] = None
        self._prev_actions: Optional[Dict[str, int]] = None

        self._ledger: Optional[LedgerWriter] = None
        if self.cfg.enabled and self.cfg.write_ledger:
            if self.cfg.ledger_path:
                ledger_path = Path(self.cfg.ledger_path)
            elif run_dir is not None:
                ledger_path = Path(run_dir) / "ledger.jsonl"
            else:
                ledger_path = Path("ledger.jsonl")
            self._ledger = LedgerWriter(ledger_path)

    # -------- pass-through attributes --------
    def __getattr__(self, item):
        return getattr(self.env, item)

    # -------- core API --------
    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        self._prev_obs = {str(k): v for k, v in obs.items() if k != "__all__"}
        self._prev_actions = None
        self._step = 0
        self._episode += 1
        return obs

    def step(self, action_dict: Dict[Any, Any]):
        # Store previous observations/actions for rule evaluation
        if self._prev_obs is None:
            # If someone called step without reset, attempt to recover.
            self.reset()

        actions = {str(k): int(v) for k, v in action_dict.items() if k != "__all__"}
        prev_obs = self._prev_obs or {}

        next_obs, rewards, dones, infos = self.env.step(action_dict)

        # Normalize keys to str for our internal bookkeeping; preserve original for env output.
        rewards_mod = dict(rewards)

        if not self.cfg.enabled:
            # Update stored obs/actions and pass through.
            self._prev_obs = {str(k): v for k, v in next_obs.items() if k != "__all__"}
            self._prev_actions = actions
            self._step += 1
            return next_obs, rewards_mod, dones, infos

        # Ensure infos is a dict of per-agent dicts
        if infos is None:
            infos = {}
        if not isinstance(infos, dict):
            infos = {}

        # Evaluate each rule per agent and apply detection+sanctions.
        for agent_id, act in actions.items():
            obs0_raw = prev_obs.get(agent_id)
            if obs0_raw is None:
                # If missing, try to fallback to next_obs (less correct, but avoids crash)
                obs0_raw = next_obs.get(agent_id)
            if obs0_raw is None:
                continue
            # SSD returns obs as {"curr_obs": ndarray, ...}; extract the array
            obs0 = extract_obs(obs0_raw)

            r0 = float(rewards.get(agent_id, 0.0))
            info0 = infos.get(agent_id, {})
            if info0 is None or not isinstance(info0, dict):
                info0 = {}

            # Per-agent aggregates over rules
            truth_any = False
            detected_any = False
            sanctions = 0
            false_positive = False
            overturned = False
            rule_records = []

            for rule in self.cfg.rules:
                truth = bool(rule.ground_truth_violation(
                    agent_id=agent_id,
                    obs=obs0,
                    action=act,
                    reward=r0,
                    info=info0,
                    env=self.env,
                ))
                truth_any = truth_any or truth

                # Noisy detection
                if truth:
                    detected = bool(self._rng.random() < float(self.cfg.p_detect_true))
                else:
                    detected = bool(self._rng.random() < float(self.cfg.p_detect_false))
                detected_any = detected_any or detected

                record = {
                    "rule": getattr(rule, "name", rule.__class__.__name__),
                    "truth": truth,
                    "detected": detected,
                }
                rule_records.append(record)

                if detected:
                    # Apply sanction for this rule detection
                    sanctions += 1
                    rewards_mod[agent_id] = float(rewards_mod.get(agent_id, 0.0)) - float(self.cfg.sanction)
                    if not truth:
                        false_positive = True

                    # Due process / review: only for false positives, and only once per timestep
                    if (not truth) and (not overturned) and (self.cfg.p_review > 0):
                        if self._rng.random() < float(self.cfg.p_review):
                            # Overturn: refund
                            rewards_mod[agent_id] = float(rewards_mod.get(agent_id, 0.0)) + float(self.cfg.sanction)
                            overturned = True

                    # Ledger: log only events (detections)
                    if self._ledger is not None:
                        self._ledger.write({
                            "episode": self._episode,
                            "t": self._step,
                            "agent_id": agent_id,
                            "rule": record["rule"],
                            "truth": truth,
                            "detected": detected,
                            "sanction": float(self.cfg.sanction),
                            "overturned": bool(overturned and (not truth)),
                            "reward_before": r0,
                            "reward_after": float(rewards_mod.get(agent_id, 0.0)),
                        })

            # Attach IML info for downstream logging
            info0.setdefault("iml", {})
            info0["iml"].update({
                "truth_any": truth_any,
                "detected_any": detected_any,
                "sanctions": sanctions,
                "false_positive": bool(false_positive),
                "overturned": bool(overturned),
                "rules": rule_records,
            })
            infos[agent_id] = info0

        # Update stored obs/actions
        self._prev_obs = {str(k): v for k, v in next_obs.items() if k != "__all__"}
        self._prev_actions = actions
        self._step += 1

        # Close ledger on episode end
        episode_done = False
        if isinstance(dones, dict):
            episode_done = bool(dones.get("__all__", False))
            if not episode_done:
                # Some envs only report per-agent dones
                # Treat as done if all known agents are done
                agent_keys = [k for k in dones.keys() if k != "__all__"]
                if agent_keys:
                    episode_done = all(bool(dones[k]) for k in agent_keys)
        if episode_done and self._ledger is not None:
            self._ledger._fh.flush()

        return next_obs, rewards_mod, dones, infos

    def close(self):
        if hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception:
                pass
        if self._ledger is not None:
            self._ledger.close()
