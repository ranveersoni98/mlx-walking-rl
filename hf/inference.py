"""Standalone play script for the BipedalWalker-v3 PPO model (numpy + mlx + gymnasium).

Usage:
    python inference.py checkpoint.npz --episodes 3
    python inference.py checkpoint.npz --episodes 1 --render
"""

import argparse

import gymnasium as gym
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten


def mlp(in_dim, hidden, out_dim):
    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=24, act_dim=4, hidden=(64, 64)):
        super().__init__()
        self.actor = mlp(obs_dim, hidden, act_dim)
        self.critic = mlp(obs_dim, hidden, 1)
        self.log_std = mx.zeros((act_dim,))

    def forward(self, obs):
        return self.actor(obs), self.log_std, self.critic(obs).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="path/local dir to .npz checkpoint")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    npz = np.load(args.checkpoint)
    env = gym.make("BipedalWalker-v3", render_mode="human" if args.render else None)
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
    params = {k[2:]: mx.array(npz[k]) for k in npz.files if k.startswith("p.")}
    model.update(tree_unflatten(params))   # type: ignore[arg-type]
    mx.eval(model.parameters())
    norm_mean, norm_var = npz["norm.mean"], npz["norm.var"]

    returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.episodes * 1000 + ep)
        ep_ret, done = 0.0, False
        while not done:
            o = ((obs - norm_mean) / np.sqrt(norm_var + 1e-8)).astype(np.float32)
            mu, _, _ = model.forward(mx.array(o[None]))
            action = np.clip(np.array(mx.clip(mu, -1.0, 1.0))[0], -1, 1).astype(np.float32)
            obs, r, term, trunc, _ = env.step(action)
            ep_ret += float(r)
            done = term or trunc
        returns.append(ep_ret)
        print(f"episode {ep + 1}: return {ep_ret:8.1f}")
    print(f"mean return: {np.mean(returns):.1f}")
    env.close()


if __name__ == "__main__":
    main()
