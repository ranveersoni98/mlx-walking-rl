"""Inspect a rollout + normalizer internals for a saved checkpoint.

Usage: uv run python scripts/inspect_rollout.py runs/walker1/ckpt_latest.npz
"""

import sys

import gymnasium as gym
import numpy as np

from mlx_walking_rl.ppo import PPOConfig, PPOTrainer


def stats(name, v):
    v = np.asarray(v)
    print(f"  {name:>10}: min {v.min():>10.4f} max {v.max():>10.4f} "
          f"mean {v.mean():>9.4f} std {v.std():>8.4f}")


def main():
    cfg = PPOConfig(run_dir="runs/debug", num_steps=512, num_envs=4)
    envs = [gym.make("BipedalWalker-v3") for _ in range(4)]
    tr = PPOTrainer(cfg, envs)
    tr.load(sys.argv[1] if len(sys.argv) > 1 else "runs/walker1/ckpt_latest.npz")
    tr.reset()

    print("== rollout with the loaded policy ==")
    batch = tr.collect_rollout()
    for k in ("obs", "act", "adv", "ret", "old_values", "old_logp"):
        stats(k, batch[k])
    print("  rew_rms.var:", tr.rew_rms.var, " count:", tr.rew_rms.count)
    print("  episode returns:", [round(x, 1) for x in tr._episode_returns[:10]])

    envs[0].close()


if __name__ == "__main__":
    main()
