from __future__ import annotations

from typing import Dict, Tuple

from .base import BasePolicy
from ..protocol import CommsAction


class PipelinePolicy(BasePolicy):
    """
    Baseline 'playbook' policy.
    - Cyber: reactive heuristics.
    - Comms: naive debunks and frequent messaging (intentionally imperfect; can violate protocol).
    """
    name = "pipeline"

    def act(self, obs: Dict, t: int) -> Tuple[str, CommsAction]:
        # --- cyber action ---
        services_down = float(obs.get("cyber_services_down", obs.get("services_down", 0.0)))
        compromised = float(obs.get("cyber_compromised_frac", obs.get("compromised_frac", 0.0)))
        detection = float(obs.get("cyber_detection_conf", obs.get("detection_conf", 0.0)))
        ransomware = float(obs.get("cyber_ransomware", 0.0))

        if ransomware > 0.5 or services_down > 0.35:
            cyber_action = "restore"
        elif compromised > 0.30:
            cyber_action = "hunt"
        elif detection < 0.35:
            cyber_action = "monitor"
        else:
            cyber_action = "patch"

        # --- comms action (naive) ---
        misbelief = float(obs.get("social_misbelief", obs.get("misbelief", 0.0)))
        trust = float(obs.get("social_trust", obs.get("trust", 0.0)))
        uncertainty = float(obs.get("social_uncertainty", obs.get("uncertainty", 0.0)))
        severity = float(obs.get("cyber_severity", obs.get("severity", 0.0)))

        # too-frequent messaging and under-evidenced debunks are common failure modes.
        if misbelief > 0.30 and severity > 0.40:
            # debunk but often without evidence/uncertainty label
            comms = CommsAction(type="debunk", target="all", evidence=False, uncertainty_label=False, intensity=0.85)
        elif uncertainty > 0.55 and trust < 0.45:
            # transparency update but forgets uncertainty label often
            comms = CommsAction(type="transparency_update", target="all", evidence=False, uncertainty_label=False, intensity=0.65)
        elif misbelief > 0.25 and trust < 0.50:
            comms = CommsAction(type="request_reports", target="all", evidence=False, uncertainty_label=True, intensity=0.70)
        else:
            comms = CommsAction(type="silence")

        return cyber_action, comms
