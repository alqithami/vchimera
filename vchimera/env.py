from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from .backends.base import CyberBackend, SocialBackend
from .coupling import CouplingBus
from .protocol import CommsAction
from .utils import StepLog


@dataclass
class EnvConfig:
    horizon: int = 60


class CyberCrisisEnv:
    """
    Coupled cyber–social environment.

    Sequence per step:
      1) social -> cyber modifiers (compliance/reporting)
      2) cyber backend step (operational action)
      3) cyber -> social narrative events
      4) social backend step (communications action)
    """

    def __init__(self, cyber: CyberBackend, social: SocialBackend, coupler: CouplingBus, cfg: EnvConfig):
        self.cyber = cyber
        self.social = social
        self.coupler = coupler
        self.cfg = cfg

        self.t = 0
        self.last_social_metrics: Dict[str, float] = {}
        self.last_cyber_obs: Dict[str, Any] = {}
        self.last_social_obs: Dict[str, Any] = {}
        self.done = False

    def reset(self, seed: int, initial_events: Dict[str, float] | None = None) -> Dict[str, Any]:
        self.t = 0
        self.done = False
        initial_events = initial_events or {}

        cyber_obs = self.cyber.reset(seed=seed)
        # initial narrative events from cyber state (e.g. early outage)
        narrative = self.coupler.narrative_events_from_cyber({
            "services_down": float(cyber_obs.get("services_down", 0.0)),
            "ransomware": float(cyber_obs.get("ransomware", 0.0)),
            "exfil_risk": float(cyber_obs.get("exfil_risk", 0.0)),
        })
        # merge any user-provided initial events
        for k, v in initial_events.items():
            narrative[k] = narrative.get(k, 0.0) + float(v)

        social_obs = self.social.reset(seed=seed, initial_events=narrative)

        self.last_social_metrics = dict(social_obs)
        self.last_cyber_obs = dict(cyber_obs)
        self.last_social_obs = dict(social_obs)

        return self._joint_obs()

    def _joint_obs(self) -> Dict[str, Any]:
        obs = {}
        obs.update({f"cyber_{k}": v for k, v in self.last_cyber_obs.items()})
        obs.update({f"social_{k}": v for k, v in self.last_social_obs.items()})
        # Provide uncluttered aliases for common fields
        obs["detection_conf"] = float(self.last_cyber_obs.get("detection_conf", 0.0))
        obs["severity"] = float(self.last_cyber_obs.get("severity", 0.0))
        obs["evidence_available"] = bool(self.last_cyber_obs.get("evidence_available", False))
        obs["misbelief"] = float(self.last_social_obs.get("misbelief", 0.0))
        obs["trust"] = float(self.last_social_obs.get("trust", 0.0))
        obs["uncertainty"] = float(self.last_social_obs.get("uncertainty", 0.0))
        obs["polarization"] = float(self.last_social_obs.get("polarization", 0.0))
        return obs

    def step(
        self,
        cyber_action: str,
        comms_action: CommsAction,
        protocol_attempted: int,
        protocol_executed: int,
        shield_interventions: int,
    ) -> Tuple[Dict[str, Any], StepLog, bool]:
        if self.done:
            return self._joint_obs(), StepLog(
                t=self.t,
                cyber_harm=0.0,
                detection_conf=float(self.last_cyber_obs.get("detection_conf", 0.0)),
                severity=float(self.last_cyber_obs.get("severity", 0.0)),
                compromised_frac=float(self.last_cyber_obs.get("compromised_frac", 0.0)),
                services_down=float(self.last_cyber_obs.get("services_down", 0.0)),
                exfil_risk=float(self.last_cyber_obs.get("exfil_risk", 0.0)),
                ransomware=float(self.last_cyber_obs.get("ransomware", 0.0)),
                misbelief=float(self.last_social_obs.get("misbelief", 0.0)),
                trust=float(self.last_social_obs.get("trust", 0.0)),
                uncertainty=float(self.last_social_obs.get("uncertainty", 0.0)),
                polarization=float(self.last_social_obs.get("polarization", 0.0)),
                protocol_attempted=protocol_attempted,
                protocol_executed=protocol_executed,
                shield_interventions=shield_interventions,
            ), True

        # social -> cyber modifiers
        modifiers = self.coupler.cyber_modifiers_from_social(self.last_social_metrics)

        cyber_obs, cyber_metrics, cyber_done = self.cyber.step(cyber_action, modifiers)

        # cyber -> social narrative events
        narrative = self.coupler.narrative_events_from_cyber(cyber_metrics)

        # official verified-update events
        vupdate = self.coupler.narrative_events_from_official_action(comms_action, cyber_obs)
        for k, v in vupdate.items():
            narrative[k] = narrative.get(k, 0.0) + float(v)

        social_obs, social_metrics = self.social.step(comms_action, narrative)

        self.last_social_metrics = dict(social_metrics)
        self.last_cyber_obs = dict(cyber_obs)
        self.last_social_obs = dict(social_obs)

        self.t += 1
        self.done = cyber_done or (self.t >= self.cfg.horizon)

        log = StepLog(
            t=self.t,
            cyber_harm=float(cyber_metrics.get("cyber_harm", 0.0)),
            detection_conf=float(cyber_metrics.get("detection_conf", 0.0)),
            severity=float(cyber_metrics.get("severity", 0.0)),
            compromised_frac=float(cyber_metrics.get("compromised_frac", 0.0)),
            services_down=float(cyber_metrics.get("services_down", 0.0)),
            exfil_risk=float(cyber_metrics.get("exfil_risk", 0.0)),
            ransomware=float(cyber_metrics.get("ransomware", 0.0)),
            misbelief=float(social_metrics.get("misbelief", 0.0)),
            trust=float(social_metrics.get("trust", 0.0)),
            uncertainty=float(social_metrics.get("uncertainty", 0.0)),
            polarization=float(social_metrics.get("polarization", 0.0)),
            protocol_attempted=int(protocol_attempted),
            protocol_executed=int(protocol_executed),
            shield_interventions=int(shield_interventions),
        )

        return self._joint_obs(), log, self.done
