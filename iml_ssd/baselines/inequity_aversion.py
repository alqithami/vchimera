"""Inequity Aversion (IA) reward wrapper.

Implements the reward modification from:
    Hughes et al., "Inequity aversion improves cooperation in intertemporal
    social dilemmas", NeurIPS 2018.

Each agent i receives a modified reward:
    r'_i = r_i - alpha * max(r_bar - r_i, 0) - beta * max(r_i - r_bar, 0)

where r_bar is the mean reward of all agents at that timestep, and:
    alpha >= 0 controls aversion to *disadvantageous* inequity (others get more)
    beta  >= 0 controls aversion to *advantageous*   inequity (I get more)

This is a *reward internalization* approach: agents' intrinsic objectives are
modified to encode social preferences.  Unlike IML, there is no external
institutional layer, no monitoring, no ledger, and no contestability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class IAConfig:
    """Configuration for the Inequity Aversion wrapper."""
    enabled: bool = False
    alpha: float = 5.0    # disadvantageous inequity aversion
    beta: float = 0.05    # advantageous inequity aversion


class IAWrapper:
    """Environment wrapper that applies inequity-averse reward shaping.

    Wraps an RLlib-style multi-agent env with the same API as IMLWrapper:
        obs_dict = env.reset()
        next_obs, rewards, dones, infos = env.step(action_dict)

    At each step, the wrapper computes the mean reward across all agents and
    adjusts each agent's reward according to the IA formula.
    """

    def __init__(self, env: Any, cfg: IAConfig, seed: Optional[int] = None):
        self.env = env
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)

    # -------- pass-through attributes --------
    def __getattr__(self, item):
        return getattr(self.env, item)

    # -------- core API --------
    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action_dict: Dict[Any, Any]):
        next_obs, rewards, dones, infos = self.env.step(action_dict)

        if not self.cfg.enabled:
            return next_obs, rewards, dones, infos

        # Compute mean reward across all agents (excluding __all__)
        agent_keys = [k for k in rewards.keys() if k != "__all__"]
        if len(agent_keys) == 0:
            return next_obs, rewards, dones, infos

        reward_vals = [float(rewards[k]) for k in agent_keys]
        r_bar = float(np.mean(reward_vals))

        rewards_mod = dict(rewards)
        for k in agent_keys:
            r_i = float(rewards[k])
            # Disadvantageous inequity: others get more than me
            disadv = max(r_bar - r_i, 0.0)
            # Advantageous inequity: I get more than others
            adv = max(r_i - r_bar, 0.0)
            r_mod = r_i - self.cfg.alpha * disadv - self.cfg.beta * adv
            rewards_mod[k] = r_mod

            # Attach IA info for downstream logging
            if infos is None:
                infos = {}
            if not isinstance(infos, dict):
                infos = {}
            info_k = infos.get(k, {})
            if info_k is None or not isinstance(info_k, dict):
                info_k = {}
            info_k["ia"] = {
                "r_original": r_i,
                "r_bar": r_bar,
                "disadv_penalty": self.cfg.alpha * disadv,
                "adv_penalty": self.cfg.beta * adv,
                "r_modified": r_mod,
            }
            infos[k] = info_k

        return next_obs, rewards_mod, dones, infos

    def close(self):
        if hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception:
                pass
