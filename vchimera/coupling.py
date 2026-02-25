from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .protocol import CommsAction


@dataclass
class CouplingConfig:
    cyber_to_social_scale: float = 1.0
    social_to_cyber_scale: float = 1.0

    outage_weight: float = 1.0
    ransomware_weight: float = 1.2
    exfil_weight: float = 0.9

    # How much verified comms events improve trust/uncertainty (handled in social backend via 'verified_update')
    verified_update_weight: float = 1.0


class CouplingBus:
    def __init__(self, cfg: CouplingConfig):
        self.cfg = cfg

    def narrative_events_from_cyber(self, cyber_metrics: Dict[str, float]) -> Dict[str, float]:
        cfg = self.cfg
        outage = float(cyber_metrics.get("services_down", 0.0))
        ransom = float(cyber_metrics.get("ransomware", 0.0))
        exfil = float(cyber_metrics.get("exfil_risk", 0.0))

        return {
            "outage": cfg.cyber_to_social_scale * cfg.outage_weight * outage,
            "ransomware": cfg.cyber_to_social_scale * cfg.ransomware_weight * ransom,
            "exfiltration": cfg.cyber_to_social_scale * cfg.exfil_weight * exfil,
            "verified_update": 0.0,
        }

    def narrative_events_from_official_action(self, a: CommsAction, cyber_obs: Dict) -> Dict[str, float]:
        cfg = self.cfg
        # verified update when evidence is included and evidence is actually available in the cyber state
        ev_avail = bool(cyber_obs.get("evidence_available", False))
        if a.type in ("transparency_update", "debunk") and a.evidence and ev_avail:
            return {"verified_update": cfg.verified_update_weight}
        return {"verified_update": 0.0}

    def cyber_modifiers_from_social(self, social_metrics: Dict[str, float]) -> Dict[str, float]:
        cfg = self.cfg
        compliance = float(social_metrics.get("compliance", 0.6))
        reporting = float(social_metrics.get("reporting", 0.6))
        # Scale around 0.5 baseline
        return {
            "compliance": 0.5 + cfg.social_to_cyber_scale * (compliance - 0.5),
            "reporting": 0.5 + cfg.social_to_cyber_scale * (reporting - 0.5),
        }
