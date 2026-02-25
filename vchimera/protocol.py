from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class CommsAction:
    """
    Structured communication action to enable verifiable protocols.

    type: one of
      - silence
      - transparency_update
      - debunk
      - prebunk
      - request_reports

    target: "all" or "comm:<k>"
    evidence: whether the message cites evidence (only meaningful for debunk/transparency_update)
    uncertainty_label: whether the message explicitly flags uncertainty
    intensity: scalar in [0,1] controlling intervention strength (volume/coverage)
    """
    type: str = "silence"
    target: str = "all"
    evidence: bool = False
    uncertainty_label: bool = False
    intensity: float = 0.5
    evidence_id: Optional[str] = None


@dataclass
class ProtocolConfig:
    cooldown_steps: int = 3
    evidence_required_for_debunk: bool = True
    evidence_detection_threshold: float = 0.6
    uncertainty_label_threshold: float = 0.55
    require_uncertainty_label_for_debunk_when_high: bool = True
    severity_gate_threshold: float = 0.75


class ProtocolEvaluator:
    """
    Evaluates and enforces a simple but explicit crisis communications protocol.
    The protocol is intended to be conservative and auditable.

    IMPORTANT: We never 'fabricate' evidence. If evidence is required but unavailable,
    we downgrade a debunk to a transparency update or silence depending on severity/cooldown.
    """

    def __init__(self, cfg: ProtocolConfig):
        self.cfg = cfg

    def evidence_available(self, obs: Dict) -> bool:
        # For simulation: evidence is available if detection confidence is high enough
        # OR if an explicit evidence flag is provided by the cyber backend (forensics completed).
        det = float(obs.get("detection_conf", 0.0))
        return bool(obs.get("evidence_available", False) or det >= self.cfg.evidence_detection_threshold)

    def count_violations(self, proposed: CommsAction, obs: Dict, t: int, last_msg_t: int) -> int:
        v = 0
        if proposed.type != "silence":
            # Cooldown
            if (t - last_msg_t) < self.cfg.cooldown_steps:
                v += 1

        if proposed.type == "debunk" and self.cfg.evidence_required_for_debunk:
            if not proposed.evidence:
                v += 1
            if not self.evidence_available(obs):
                v += 1

        # Uncertainty labeling rule
        unc = float(obs.get("uncertainty", 0.0))
        if proposed.type == "transparency_update" and unc >= self.cfg.uncertainty_label_threshold:
            if not proposed.uncertainty_label:
                v += 1

        if proposed.type == "debunk" and self.cfg.require_uncertainty_label_for_debunk_when_high:
            if unc >= self.cfg.uncertainty_label_threshold and not proposed.uncertainty_label:
                v += 1

        # Severity gate: in high severity but low evidence, do not debunk.
        sev = float(obs.get("severity", 0.0))
        det = float(obs.get("detection_conf", 0.0))
        if proposed.type == "debunk" and sev >= self.cfg.severity_gate_threshold and det < self.cfg.evidence_detection_threshold:
            v += 1

        return v

    def shield(self, proposed: CommsAction, obs: Dict, t: int, last_msg_t: int) -> Tuple[CommsAction, int]:
        """
        Returns a protocol-compliant action and the number of interventions applied.
        """
        cfg = self.cfg
        interventions = 0
        a = proposed

        # 1) Cooldown is hard: if violated, silence.
        if a.type != "silence" and (t - last_msg_t) < cfg.cooldown_steps:
            interventions += 1
            return CommsAction(type="silence"), interventions

        unc = float(obs.get("uncertainty", 0.0))
        sev = float(obs.get("severity", 0.0))
        det = float(obs.get("detection_conf", 0.0))
        ev_avail = self.evidence_available(obs)

        # 2) Severity gate (no debunk during high severity & low verification)
        if a.type == "debunk" and sev >= cfg.severity_gate_threshold and det < cfg.evidence_detection_threshold:
            interventions += 1
            a = CommsAction(
                type="transparency_update",
                target=a.target,
                evidence=ev_avail,  # only true if available
                uncertainty_label=True,
                intensity=max(0.5, a.intensity),
                evidence_id=a.evidence_id if ev_avail else None,
            )

        # 3) Evidence for debunks (never fabricate)
        if a.type == "debunk" and cfg.evidence_required_for_debunk:
            if (not a.evidence) or (not ev_avail):
                interventions += 1
                # downgrade to transparency update (or request_reports if no evidence and high uncertainty)
                if not ev_avail and unc >= cfg.uncertainty_label_threshold:
                    a = CommsAction(type="request_reports", target="all", evidence=False, uncertainty_label=True, intensity=0.7)
                else:
                    a = CommsAction(
                        type="transparency_update",
                        target=a.target,
                        evidence=ev_avail,
                        uncertainty_label=(unc >= cfg.uncertainty_label_threshold) or a.uncertainty_label,
                        intensity=max(0.5, a.intensity),
                        evidence_id=a.evidence_id if ev_avail else None,
                    )

        # 4) Uncertainty labeling requirement
        if a.type in ("transparency_update", "debunk"):
            if unc >= cfg.uncertainty_label_threshold and not a.uncertainty_label:
                interventions += 1
                a = replace(a, uncertainty_label=True)

        # Final pass: ensure evidence_id is None if evidence is false
        if not a.evidence and a.evidence_id is not None:
            interventions += 1
            a = replace(a, evidence_id=None)

        return a, interventions
