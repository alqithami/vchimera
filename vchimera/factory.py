from __future__ import annotations

from typing import Any, Dict, Tuple

import yaml

from .backends.cyber_attackgraph import AttackGraphConfig, AttackGraphCyberBackend
from .backends.social_abm import PlatformConfig, SocialABMConfig, SocialABMBackend
from .coupling import CouplingBus, CouplingConfig
from .env import CyberCrisisEnv, EnvConfig
from .protocol import ProtocolConfig, ProtocolEvaluator


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def build_env(scenario_cfg: Dict[str, Any]) -> Tuple[CyberCrisisEnv, ProtocolEvaluator, Dict[str, Any]]:
    """
    Builds environment + protocol evaluator from a scenario config dict.

    Cyber backend selection:
      - cyber.backend: "attackgraph" (default) or "cyborg" (optional dependency)
    """
    cyber_raw = dict(scenario_cfg.get("cyber", {}))
    backend_name = str(cyber_raw.pop("backend", "attackgraph")).lower().strip()

    if backend_name == "attackgraph":
        cyber_cfg = AttackGraphConfig(**cyber_raw)
        cyber = AttackGraphCyberBackend(cyber_cfg)
    elif backend_name == "cyborg":
        # Optional adapter
        from .backends.adapters.cyborg_adapter import CybORGAdapter, CybORGAdapterConfig

        cyber_cfg = CybORGAdapterConfig(**cyber_raw)
        cyber = CybORGAdapter(cyber_cfg)
    else:
        raise ValueError(f"Unknown cyber backend: {backend_name}")

    social_cfg_raw = scenario_cfg.get("social", {})
    plats = []
    for p in social_cfg_raw.get("platforms", []):
        plats.append(PlatformConfig(**p))
    social_cfg = SocialABMConfig(**{k: v for k, v in social_cfg_raw.items() if k != "platforms"})
    social_cfg.platforms = plats

    coupling_cfg = CouplingConfig(**scenario_cfg.get("coupling", {}))
    env_cfg = EnvConfig(**scenario_cfg.get("env", {}))
    protocol_cfg = ProtocolConfig(**scenario_cfg.get("protocol", {}))

    social = SocialABMBackend(social_cfg)
    coupler = CouplingBus(coupling_cfg)
    env = CyberCrisisEnv(cyber=cyber, social=social, coupler=coupler, cfg=env_cfg)

    evaluator = ProtocolEvaluator(protocol_cfg)
    info = {
        "cyber_backend": backend_name,
        "cyber_cfg": cyber_cfg,
        "social_cfg": social_cfg,
        "coupling_cfg": coupling_cfg,
        "env_cfg": env_cfg,
        "protocol_cfg": protocol_cfg,
    }
    return env, evaluator, info
