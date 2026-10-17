"""Actor-critic networks for continuous-control PPO, implemented with MLX."""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

LOG_2PI = math.log(2.0 * math.pi)
# 0.5 * log(2 * pi * e) -> per-dim Gaussian entropy constant
ENTROPY_CONST = 0.5 * math.log(2.0 * math.pi * math.e)


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Independent actor/critic MLPs; state-independent log-std for the Gaussian policy.

    forward(obs) -> (mu [B, A], log_std [A], value [B])
    """

    def __init__(
        self,
        obs_dim: int = 24,
        act_dim: int = 4,
        hidden: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        self.actor = _mlp(obs_dim, hidden, act_dim)
        self.critic = _mlp(obs_dim, hidden, 1)
        self.log_std = mx.zeros((act_dim,))  # trainable parameter, initial std = 1

    def forward(self, obs: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        mu = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return mu, self.log_std, value

    @staticmethod
    def log_prob(mu: mx.array, log_std: mx.array, x: mx.array) -> mx.array:
        """Gaussian log-prob of actions ``x`` given per-sample ``mu`` and shared ``log_std``.

        ``log_std`` may be shape [A]; it broadcasts against mu [B, A]. Returns [B].
        """
        return (-0.5 * (mx.square((x - mu) / mx.exp(log_std)) + 2.0 * log_std + LOG_2PI)).sum(-1)

    @staticmethod
    def entropy(log_std: mx.array) -> mx.array:
        """Diagonal Gaussian entropy, scalar when ``log_std`` is [A]."""
        return (log_std + ENTROPY_CONST).sum(-1).mean()
