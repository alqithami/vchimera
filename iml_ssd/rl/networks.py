from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _orthogonal_init(layer: nn.Module, gain: float = 1.0) -> None:
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)


class SharedCNNActorCritic(nn.Module):
    def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int):
        super().__init__()
        h, w, c = obs_shape
        self.obs_shape = obs_shape
        self.n_actions = int(n_actions)

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Determine flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flat = self.cnn(dummy).shape[-1]

        self.mlp = nn.Sequential(
            nn.Linear(n_flat, 256),
            nn.ReLU(),
        )
        self.pi = nn.Linear(256, self.n_actions)
        self.v = nn.Linear(256, 1)

        # Init
        for m in self.modules():
            _orthogonal_init(m, gain=1.0)
        _orthogonal_init(self.pi, gain=0.01)
        _orthogonal_init(self.v, gain=1.0)

    def forward(self, obs: torch.Tensor):
        """obs: (B,H,W,C) or (B,C,H,W)"""
        if obs.dim() != 4:
            raise ValueError(f"Expected obs with 4 dims, got {obs.shape}")
        if obs.shape[1:] == torch.Size(self.obs_shape):
            # (B,H,W,C) -> (B,C,H,W)
            obs = obs.permute(0, 3, 1, 2)
        # else assume already (B,C,H,W)
        z = self.cnn(obs)
        z = self.mlp(z)
        logits = self.pi(z)
        value = self.v(z).squeeze(-1)
        return logits, value

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, entropy, value

    def get_value(self, obs: torch.Tensor):
        _, value = self.forward(obs)
        return value
