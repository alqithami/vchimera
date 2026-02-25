from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class CyberBackend(ABC):
    @abstractmethod
    def reset(self, seed: int) -> Dict[str, Any]:
        ...

    @abstractmethod
    def step(self, action: str, modifiers: Dict[str, float]) -> Tuple[Dict[str, Any], Dict[str, float], bool]:
        """
        Returns (obs, metrics, done).
        metrics should include:
          - cyber_harm
          - detection_conf
          - severity
          - compromised_frac
          - services_down
          - exfil_risk
          - ransomware
          - evidence_available (bool)
        """
        ...


class SocialBackend(ABC):
    @abstractmethod
    def reset(self, seed: int, initial_events: Dict[str, float]) -> Dict[str, Any]:
        ...

    @abstractmethod
    def step(self, comms_action: Any, narrative_events: Dict[str, float]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Returns (obs, metrics).
        metrics should include:
          - misbelief
          - trust
          - uncertainty
          - polarization
          - compliance
          - reporting
        """
        ...
