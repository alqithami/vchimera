from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import SharedCNNActorCritic


@dataclass
class PPOConfig:
    total_steps: int = 2_000_000
    rollout_steps: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2.5e-4
    num_epochs: int = 4
    minibatch_size: int = 512
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    device: str = "auto"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


class RolloutBuffer:
    def __init__(self, rollout_steps: int, num_agents: int, obs_shape: Tuple[int, int, int], device: torch.device):
        self.T = int(rollout_steps)
        self.N = int(num_agents)
        h, w, c = obs_shape
        self.obs = torch.zeros((self.T, self.N, h, w, c), dtype=torch.float32, device=device)
        self.actions = torch.zeros((self.T, self.N), dtype=torch.int64, device=device)
        self.logprobs = torch.zeros((self.T, self.N), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((self.T, self.N), dtype=torch.float32, device=device)
        self.dones = torch.zeros((self.T, self.N), dtype=torch.float32, device=device)
        self.values = torch.zeros((self.T, self.N), dtype=torch.float32, device=device)

        self.ptr = 0

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        t = self.ptr
        self.obs[t].copy_(obs)
        self.actions[t].copy_(actions)
        self.logprobs[t].copy_(logprobs)
        self.rewards[t].copy_(rewards)
        self.dones[t].copy_(dones)
        self.values[t].copy_(values)
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.T


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute advantages and returns.

    rewards, dones, values: (T,N)
    last_values: (N,)
    """
    T, N = rewards.shape
    advantages = torch.zeros((T, N), dtype=torch.float32, device=rewards.device)
    last_gae = torch.zeros((N,), dtype=torch.float32, device=rewards.device)

    for t in reversed(range(T)):
        next_values = last_values if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def ppo_update(
    model: SharedCNNActorCritic,
    optimizer: optim.Optimizer,
    buf: RolloutBuffer,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    cfg: PPOConfig,
) -> Dict[str, float]:
    device = advantages.device
    T, N = buf.rewards.shape
    B = T * N

    # Flatten
    b_obs = buf.obs.reshape(B, *buf.obs.shape[2:])
    b_actions = buf.actions.reshape(B)
    b_logprobs = buf.logprobs.reshape(B)
    b_adv = advantages.reshape(B)
    b_returns = returns.reshape(B)
    b_values = buf.values.reshape(B)

    # Normalize advantages
    b_adv = (b_adv - b_adv.mean()) / (b_adv.std(unbiased=False) + 1e-8)

    idx = torch.randperm(B, device=device)

    pg_losses = []
    v_losses = []
    ent_losses = []
    clipfracs = []
    approx_kls = []

    for epoch in range(cfg.num_epochs):
        for start in range(0, B, cfg.minibatch_size):
            mb_idx = idx[start : start + cfg.minibatch_size]
            obs_mb = b_obs[mb_idx]
            actions_mb = b_actions[mb_idx]
            old_logprob_mb = b_logprobs[mb_idx]
            adv_mb = b_adv[mb_idx]
            returns_mb = b_returns[mb_idx]
            old_values_mb = b_values[mb_idx]

            _, new_logprob, entropy, new_values = model.get_action_and_value(obs_mb, action=actions_mb)

            logratio = new_logprob - old_logprob_mb
            ratio = logratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())
                approx_kls.append(approx_kl.item())

            # Policy loss
            pg_loss1 = -adv_mb * ratio
            pg_loss2 = -adv_mb * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss (clipped)
            new_values = new_values.view(-1)
            v_loss_unclipped = (new_values - returns_mb) ** 2
            v_clipped = old_values_mb + torch.clamp(new_values - old_values_mb, -cfg.clip_coef, cfg.clip_coef)
            v_loss_clipped = (v_clipped - returns_mb) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()

            loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            pg_losses.append(pg_loss.item())
            v_losses.append(v_loss.item())
            ent_losses.append(entropy_loss.item())

        if cfg.target_kl is not None and np.mean(approx_kls) > cfg.target_kl:
            break

    return {
        "loss/policy": float(np.mean(pg_losses)) if pg_losses else 0.0,
        "loss/value": float(np.mean(v_losses)) if v_losses else 0.0,
        "loss/entropy": float(np.mean(ent_losses)) if ent_losses else 0.0,
        "diagnostics/clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
        "diagnostics/approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
    }
