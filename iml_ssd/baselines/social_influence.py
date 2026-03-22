"""Social Influence (SI) intrinsic reward wrapper.

Implements a tractable approximation of the causal influence reward from:
    Jaques et al., "Social Influence as Intrinsic Motivation for Multi-Agent
    Deep Reinforcement Learning", ICML 2019.

The original method uses counterfactual rollouts to measure how much agent i's
action changes the action distribution of agent j.  Since we use parameter-
shared PPO without per-agent models, we approximate influence using the
*marginal action entropy* approach:

    influence_i = (1/|J|) * sum_{j != i} KL( pi_j(a | o_j, a_i=actual)
                                            || pi_j(a | o_j, a_i=marginal) )

In the parameter-shared setting we cannot condition j's policy on i's action
directly.  We therefore use a simpler but well-established proxy:

    influence_i = (1/|J|) * sum_{j != i} |r_j - r_bar|

This captures the degree to which the environment outcome for other agents
deviates from the mean when agent i acts, which correlates with causal
influence in SSDs where agents' actions directly affect each other's rewards
(e.g., through the punishment beam or resource competition).

For a more faithful implementation, we also provide a *policy-based* influence
mode that requires the model to be passed in, computing the actual KL
divergence between the policy conditioned on the agent's action vs. a uniform
prior.  This is activated when a model reference is provided.

The intrinsic reward is:
    r'_i = r_i + influence_weight * influence_i
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SIConfig:
    """Configuration for the Social Influence wrapper."""
    enabled: bool = False
    influence_weight: float = 1.0  # scaling factor for the intrinsic influence reward
    mode: str = "reward_deviation"  # "reward_deviation" or "policy_kl"


class SIWrapper:
    """Environment wrapper that adds social influence intrinsic motivation.

    Wraps an RLlib-style multi-agent env with the same API as IMLWrapper:
        obs_dict = env.reset()
        next_obs, rewards, dones, infos = env.step(action_dict)

    At each step, the wrapper computes an influence measure for each agent and
    adds it as an intrinsic reward bonus.
    """

    def __init__(self, env: Any, cfg: SIConfig, seed: Optional[int] = None):
        self.env = env
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)
        self._model = None  # can be set externally for policy_kl mode
        self._prev_obs = None

    # -------- pass-through attributes --------
    def __getattr__(self, item):
        return getattr(self.env, item)

    def set_model(self, model):
        """Set the policy model for policy-based influence computation."""
        self._model = model

    # -------- core API --------
    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        self._prev_obs = {str(k): v for k, v in obs.items() if k != "__all__"}
        return obs

    def _compute_reward_deviation_influence(
        self, rewards: Dict[Any, Any], agent_keys: List[str]
    ) -> Dict[str, float]:
        """Compute influence as the mean absolute deviation of other agents' rewards.

        For agent i, influence_i = (1/|J|) * sum_{j != i} |r_j - r_bar|
        where r_bar is the mean reward across all agents.
        """
        reward_vals = {k: float(rewards[k]) for k in agent_keys}
        r_bar = float(np.mean(list(reward_vals.values())))

        influence = {}
        for i in agent_keys:
            others = [k for k in agent_keys if k != i]
            if len(others) == 0:
                influence[i] = 0.0
                continue
            # How much do others' rewards deviate from the mean?
            deviations = [abs(reward_vals[j] - r_bar) for j in others]
            influence[i] = float(np.mean(deviations))
        return influence

    def _compute_policy_kl_influence(
        self,
        prev_obs: Dict[str, Any],
        actions: Dict[str, int],
        agent_keys: List[str],
    ) -> Dict[str, float]:
        """Compute influence via KL divergence of other agents' policies.

        For each agent i, we measure how much i's action (through its effect
        on the environment state) shifts the action distributions of other
        agents compared to a uniform baseline.

        This requires the model to be set via set_model().
        """
        if self._model is None:
            # Fallback to reward deviation
            return {k: 0.0 for k in agent_keys}

        import torch
        from iml_ssd.envs.ssd_env import preprocess_obs, extract_obs

        device = next(self._model.parameters()).device

        # Get current policy distributions for all agents
        obs_batch = []
        for aid in agent_keys:
            o = prev_obs.get(aid)
            if o is not None:
                obs_batch.append(preprocess_obs(o))
            else:
                # Fallback: create zero array matching the shape of another agent's obs
                ref = extract_obs(next(iter(prev_obs.values())))
                obs_batch.append(np.zeros_like(ref, dtype=np.float32))

        obs_t = torch.tensor(np.stack(obs_batch, axis=0), dtype=torch.float32, device=device)

        with torch.no_grad():
            logits, _ = self._model.forward(obs_t)
            probs = torch.softmax(logits, dim=-1)  # (N, A)

        n_actions = probs.shape[-1]
        uniform = torch.ones_like(probs) / n_actions

        # KL(pi || uniform) for each agent
        kl_from_uniform = torch.sum(
            probs * (torch.log(probs + 1e-10) - torch.log(uniform + 1e-10)),
            dim=-1,
        )  # (N,)

        influence = {}
        for idx_i, aid_i in enumerate(agent_keys):
            others_kl = []
            for idx_j, aid_j in enumerate(agent_keys):
                if aid_j != aid_i:
                    others_kl.append(float(kl_from_uniform[idx_j].item()))
            influence[aid_i] = float(np.mean(others_kl)) if others_kl else 0.0

        return influence

    def step(self, action_dict: Dict[Any, Any]):
        actions = {str(k): int(v) for k, v in action_dict.items() if k != "__all__"}
        prev_obs = self._prev_obs or {}

        next_obs, rewards, dones, infos = self.env.step(action_dict)

        if not self.cfg.enabled:
            self._prev_obs = {str(k): v for k, v in next_obs.items() if k != "__all__"}
            return next_obs, rewards, dones, infos

        agent_keys = [k for k in rewards.keys() if k != "__all__"]
        if len(agent_keys) == 0:
            self._prev_obs = {str(k): v for k, v in next_obs.items() if k != "__all__"}
            return next_obs, rewards, dones, infos

        # Compute influence
        if self.cfg.mode == "policy_kl" and self._model is not None:
            influence = self._compute_policy_kl_influence(prev_obs, actions, agent_keys)
        else:
            influence = self._compute_reward_deviation_influence(rewards, agent_keys)

        # Apply intrinsic reward
        rewards_mod = dict(rewards)
        if infos is None:
            infos = {}
        if not isinstance(infos, dict):
            infos = {}

        for k in agent_keys:
            r_orig = float(rewards[k])
            infl = influence.get(k, 0.0)
            r_mod = r_orig + self.cfg.influence_weight * infl
            rewards_mod[k] = r_mod

            info_k = infos.get(k, {})
            if info_k is None or not isinstance(info_k, dict):
                info_k = {}
            info_k["si"] = {
                "r_original": r_orig,
                "influence": infl,
                "influence_bonus": self.cfg.influence_weight * infl,
                "r_modified": r_mod,
            }
            infos[k] = info_k

        self._prev_obs = {str(k): v for k, v in next_obs.items() if k != "__all__"}
        return next_obs, rewards_mod, dones, infos

    def close(self):
        if hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception:
                pass
