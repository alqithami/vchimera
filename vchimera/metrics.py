from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .utils import auc, StepLog


@dataclass
class EpisodeSummary:
    cyber_harm_auc: float
    misbelief_auc: float
    trust_auc: float
    uncertainty_auc: float
    polarization_auc: float

    protocol_attempted: int
    protocol_executed: int
    shield_interventions: int

    final_cyber_harm: float
    final_misbelief: float
    final_trust: float

    # additional diagnostics
    exfil_risk_auc: float
    services_down_auc: float
    compromised_auc: float


def summarize_episode(steps: List[StepLog]) -> EpisodeSummary:
    if len(steps) == 0:
        return EpisodeSummary(
            cyber_harm_auc=0.0,
            misbelief_auc=0.0,
            trust_auc=0.0,
            uncertainty_auc=0.0,
            polarization_auc=0.0,
            protocol_attempted=0,
            protocol_executed=0,
            shield_interventions=0,
            final_cyber_harm=0.0,
            final_misbelief=0.0,
            final_trust=0.0,
            exfil_risk_auc=0.0,
            services_down_auc=0.0,
            compromised_auc=0.0,
        )

    cyber_harm_auc = auc([s.cyber_harm for s in steps])
    misbelief_auc = auc([s.misbelief for s in steps])
    trust_auc = auc([s.trust for s in steps])
    uncertainty_auc = auc([s.uncertainty for s in steps])
    polarization_auc = auc([s.polarization for s in steps])

    protocol_attempted = int(sum(s.protocol_attempted for s in steps))
    protocol_executed = int(sum(s.protocol_executed for s in steps))
    shield_interventions = int(sum(s.shield_interventions for s in steps))

    exfil_risk_auc = auc([s.exfil_risk for s in steps])
    services_down_auc = auc([s.services_down for s in steps])
    compromised_auc = auc([s.compromised_frac for s in steps])

    last = steps[-1]
    return EpisodeSummary(
        cyber_harm_auc=float(cyber_harm_auc),
        misbelief_auc=float(misbelief_auc),
        trust_auc=float(trust_auc),
        uncertainty_auc=float(uncertainty_auc),
        polarization_auc=float(polarization_auc),
        protocol_attempted=protocol_attempted,
        protocol_executed=protocol_executed,
        shield_interventions=shield_interventions,
        final_cyber_harm=float(last.cyber_harm),
        final_misbelief=float(last.misbelief),
        final_trust=float(last.trust),
        exfil_risk_auc=float(exfil_risk_auc),
        services_down_auc=float(services_down_auc),
        compromised_auc=float(compromised_auc),
    )


def summary_to_row(summary: EpisodeSummary) -> Dict[str, float]:
    return {
        "cyber_harm_auc": summary.cyber_harm_auc,
        "misbelief_auc": summary.misbelief_auc,
        "trust_auc": summary.trust_auc,
        "uncertainty_auc": summary.uncertainty_auc,
        "polarization_auc": summary.polarization_auc,
        "protocol_attempted": float(summary.protocol_attempted),
        "protocol_executed": float(summary.protocol_executed),
        "shield_interventions": float(summary.shield_interventions),
        "final_cyber_harm": summary.final_cyber_harm,
        "final_misbelief": summary.final_misbelief,
        "final_trust": summary.final_trust,
        "exfil_risk_auc": summary.exfil_risk_auc,
        "services_down_auc": summary.services_down_auc,
        "compromised_auc": summary.compromised_auc,
    }
