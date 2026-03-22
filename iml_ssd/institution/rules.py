from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


class NormRule:
    """Interface for a normative rule.

    The rule returns whether a *ground-truth* violation occurred for a given agent step.
    """

    name: str = "rule"

    def ground_truth_violation(
        self,
        *,
        agent_id: str,
        obs: np.ndarray,
        action: int,
        reward: float,
        info: Dict[str, Any],
        env: Any,
    ) -> bool:
        raise NotImplementedError


@dataclass
class NoPunishmentBeamRule(NormRule):
    """Violation if the agent fires the punishment beam ("FIRE" in many SSD configs).

    Detection is done using the action id if available, otherwise using the characteristic
    fire cost in SSD (commonly -1 for the shooter), as described in several SSD papers.
    """

    name: str = "no_punishment_beam"
    fire_action_name: str = "FIRE"
    # Fallback heuristic: in SSD Harvest/Cleanup, firing beam typically costs the shooter -1
    fire_cost: float = -1.0
    # How close the reward must be to be considered a "fire" event (float safety).
    cost_tol: float = 1e-6

    def _get_fire_action_id(self, env: Any) -> Optional[int]:
        # Try common locations for action dictionaries
        for attr in ("all_actions", "ACTIONS", "actions", "action_map"):
            if hasattr(env, attr):
                d = getattr(env, attr)
                if isinstance(d, dict) and self.fire_action_name in d:
                    try:
                        return int(d[self.fire_action_name])
                    except Exception:
                        pass
        return None

    def ground_truth_violation(
        self,
        *,
        agent_id: str,
        obs: np.ndarray,
        action: int,
        reward: float,
        info: Dict[str, Any],
        env: Any,
    ) -> bool:
        fire_id = self._get_fire_action_id(env)
        if fire_id is not None:
            return int(action) == int(fire_id)
        # Fallback: reward is the shooter cost
        return abs(float(reward) - float(self.fire_cost)) <= self.cost_tol


@dataclass
class LowAppleDensityHarvestRule(NormRule):
    """Violation if an agent collects an apple when its local apple density is below a threshold.

    This is a *proxy* rule for "sustainable harvesting" that can be tuned/abated.

    Ground-truth 'collected apple' is approximated as reward >= apple_reward (default 1.0).
    Local apple density is computed from the agent's observation via a simple RGB heuristic:
    apples are typically bright green pixels in SSD renderings.

    IMPORTANT: This rule is OFF by default in the provided configs (use with care).
    """

    name: str = "low_density_harvest"
    apple_reward: float = 1.0
    density_threshold: float = 0.02  # fraction of pixels that look like apples
    # RGB heuristics for apple pixels (expects obs in 0..1 or 0..255)
    green_min: float = 0.75  # if normalized
    red_max: float = 0.35
    blue_max: float = 0.35

    def _to01(self, obs: np.ndarray) -> np.ndarray:
        if obs.dtype != np.float32:
            obs = obs.astype(np.float32)
        if obs.max() > 1.5:
            obs = obs / 255.0
        return obs

    def _apple_mask(self, obs01: np.ndarray) -> np.ndarray:
        # obs01 shape: HxWx3
        r = obs01[..., 0]
        g = obs01[..., 1]
        b = obs01[..., 2]
        return (g >= self.green_min) & (r <= self.red_max) & (b <= self.blue_max)

    def ground_truth_violation(
        self,
        *,
        agent_id: str,
        obs: np.ndarray,
        action: int,
        reward: float,
        info: Dict[str, Any],
        env: Any,
    ) -> bool:
        if float(reward) < float(self.apple_reward):
            return False
        obs01 = self._to01(obs)
        mask = self._apple_mask(obs01)
        density = float(mask.mean())
        return density < float(self.density_threshold)


@dataclass
class HighWasteNoCleanRule(NormRule):
    """Violation if waste appears high in the agent's observation and the agent does not clean.

    This is a proxy rule for public-good contribution in Cleanup.

    IMPORTANT: This rule is OFF by default in the provided configs (use with care).
    """

    name: str = "high_waste_no_clean"
    clean_action_name: str = "CLEAN"
    waste_frac_threshold: float = 0.05
    # Approximate waste pixels as gray-ish in RGB
    gray_tol: float = 0.06  # in 0..1 space
    min_intensity: float = 0.15
    max_intensity: float = 0.85

    def _to01(self, obs: np.ndarray) -> np.ndarray:
        if obs.dtype != np.float32:
            obs = obs.astype(np.float32)
        if obs.max() > 1.5:
            obs = obs / 255.0
        return obs

    def _get_clean_action_id(self, env: Any) -> Optional[int]:
        for attr in ("all_actions", "ACTIONS", "actions", "action_map"):
            if hasattr(env, attr):
                d = getattr(env, attr)
                if isinstance(d, dict) and self.clean_action_name in d:
                    try:
                        return int(d[self.clean_action_name])
                    except Exception:
                        pass
        return None

    def _waste_mask(self, obs01: np.ndarray) -> np.ndarray:
        r = obs01[..., 0]
        g = obs01[..., 1]
        b = obs01[..., 2]
        # gray-ish
        grayish = (np.abs(r - g) <= self.gray_tol) & (np.abs(g - b) <= self.gray_tol)
        intensity = (r + g + b) / 3.0
        in_range = (intensity >= self.min_intensity) & (intensity <= self.max_intensity)
        return grayish & in_range

    def ground_truth_violation(
        self,
        *,
        agent_id: str,
        obs: np.ndarray,
        action: int,
        reward: float,
        info: Dict[str, Any],
        env: Any,
    ) -> bool:
        obs01 = self._to01(obs)
        waste_frac = float(self._waste_mask(obs01).mean())
        if waste_frac < float(self.waste_frac_threshold):
            return False

        clean_id = self._get_clean_action_id(env)
        if clean_id is None:
            # If we cannot resolve CLEAN action id, we cannot enforce this rule.
            return False
        return int(action) != int(clean_id)
