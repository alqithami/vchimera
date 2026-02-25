from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

from .base import BasePolicy
from ..protocol import CommsAction


@dataclass
class VChimeraConfig:
    use_coupling: bool = True
    use_targeting: bool = True

    # thresholds
    misbelief_high: float = 0.30
    misbelief_mid: float = 0.22
    trust_low: float = 0.45
    uncertainty_high: float = 0.55
    severity_high: float = 0.70
    severity_mid: float = 0.45
    detection_good: float = 0.60


class VChimeraPolicy(BasePolicy):
    """
    Coupling-aware, protocol-aware decision policy (pre-shield).
    The protocol shield is applied externally in the runner.
    """
    name = "vchimera"

    def __init__(self, cfg: Optional[VChimeraConfig] = None):
        self.cfg = cfg or VChimeraConfig()

    def _pick_target(self, obs: Dict) -> str:
        if not self.cfg.use_targeting:
            return "all"
        comm = obs.get("social_comm_misbelief") or obs.get("comm_misbelief")
        if comm is None:
            return "all"
        try:
            arr = np.array(comm, dtype=float)
            if arr.size == 0 or np.all(np.isnan(arr)):
                return "all"
            k = int(np.nanargmax(arr))
            return f"comm:{k}"
        except Exception:
            return "all"

    def act(self, obs: Dict, t: int) -> Tuple[str, CommsAction]:
        cfg = self.cfg

        # key state
        detection = float(obs.get("cyber_detection_conf", obs.get("detection_conf", 0.0)))
        severity = float(obs.get("cyber_severity", obs.get("severity", 0.0)))
        compromised = float(obs.get("cyber_compromised_frac", obs.get("compromised_frac", 0.0)))
        services_down = float(obs.get("cyber_services_down", obs.get("services_down", 0.0)))
        evidence_avail = bool(obs.get("cyber_evidence_available", obs.get("evidence_available", False)))

        misbelief = float(obs.get("social_misbelief", obs.get("misbelief", 0.0)))
        trust = float(obs.get("social_trust", obs.get("trust", 0.0)))
        uncertainty = float(obs.get("social_uncertainty", obs.get("uncertainty", 0.0)))
        compliance = float(obs.get("social_compliance", obs.get("compliance", 0.6)))
        reporting = float(obs.get("social_reporting", obs.get("reporting", 0.6)))

        # --- cyber action ---
        if cfg.use_coupling:
            # when social reporting/compliance are weak, prioritize monitoring/hunting to raise certainty/evidence
            if severity >= cfg.severity_high and (not evidence_avail) and detection < cfg.detection_good:
                cyber_action = "monitor" if reporting >= 0.45 else "hunt"
            elif compromised > 0.30:
                cyber_action = "hunt"
            elif services_down > 0.35:
                cyber_action = "restore"
            else:
                # if compliance low, patch is less effective; build detection first
                cyber_action = "monitor" if compliance < 0.45 else "patch"
        else:
            # no-coupling ablation: ignore social modifiers
            if compromised > 0.30:
                cyber_action = "hunt"
            elif services_down > 0.35:
                cyber_action = "restore"
            elif detection < 0.40:
                cyber_action = "monitor"
            else:
                cyber_action = "patch"

        # --- comms action (protocol-aware intent; shield enforces strictly) ---
        target = self._pick_target(obs)

        if severity >= cfg.severity_high and (not evidence_avail):
            # high-severity, low-verification => transparency with uncertainty label; request reports if reporting low
            if reporting < 0.45 and t % 4 == 0:
                comms = CommsAction(type="request_reports", target="all", evidence=False, uncertainty_label=True, intensity=0.75)
            else:
                comms = CommsAction(type="transparency_update", target="all", evidence=False, uncertainty_label=True, intensity=0.70)

        elif misbelief >= cfg.misbelief_high and evidence_avail and detection >= cfg.detection_good:
            # targeted evidence-based debunk
            comms = CommsAction(type="debunk", target=target, evidence=True, uncertainty_label=(uncertainty >= cfg.uncertainty_high), intensity=0.75, evidence_id="forensics:verified")

        elif uncertainty >= cfg.uncertainty_high and trust <= cfg.trust_low:
            # restore trust with transparent update
            comms = CommsAction(type="transparency_update", target="all", evidence=evidence_avail, uncertainty_label=True, intensity=0.65, evidence_id=("status:verified" if evidence_avail else None))

        elif (t <= 6) and misbelief >= cfg.misbelief_mid:
            # early prebunk to increase resilience in at-risk communities
            comms = CommsAction(type="prebunk", target=target, evidence=False, uncertainty_label=True, intensity=0.60)

        elif misbelief >= cfg.misbelief_mid and evidence_avail and detection >= 0.50:
            # mid-stage: limited debunk, lower intensity
            comms = CommsAction(type="debunk", target=target, evidence=True, uncertainty_label=True, intensity=0.55, evidence_id="forensics:partial")

        else:
            comms = CommsAction(type="silence")

        return cyber_action, comms
