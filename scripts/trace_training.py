"""Per-iteration trace of V vs GAE targets during training — the divergence detector.

Should show targets flat, V tracking them, vf_loss small. If targets drift and V
lags, something in the GAE/bootstrap/normalizer chain is broken again.

Usage: uv run python scripts/trace_training.py
"""

import gymnasium as gym
import mlx.core as mx
import numpy as np

from mlx_walking_rl.ppo import PPOConfig, PPOTrainer


def main():
    cfg = PPOConfig(run_dir="runs/debug", num_steps=512, num_envs=4,
                    epochs=2, minibatch_size=512, anneal_lr=False,
                    total_steps=10 ** 9)
    envs = [gym.make("BipedalWalker-v3") for _ in range(4)]
    tr = PPOTrainer(cfg, envs)
    tr.reset()

    def snap(tag, batch):
        x = mx.array(batch["obs"][:512])
        _, _, v = tr.model.forward(x)
        v = np.array(mx.eval(v) or v)
        print(f"    {tag}: V mean {v.mean():7.3f} std {v.std():6.3f} | "
              f"targets mean {batch['ret'].mean():7.3f} std {batch['ret'].std():6.3f} | "
              f"adv mean {batch['adv'].mean():7.3f} std {batch['adv'].std():6.3f} | "
              f"rew_var {tr.rew_rms.var[0]:9.1f}")

    for it in range(1, 8):
        batch = tr.collect_rollout()
        snap(f"it {it} collected", batch)
        stats = tr.update(batch)
        snap(f"it {it} UPDATED  ", batch)
        print(f"    vf_loss {stats['vf_loss']:10.4f}  pi {stats['pi_loss']:9.5f} "
              f"kl {stats['approx_kl']:9.6f} ent {stats['entropy']:6.3f}")

    envs[0].close()


if __name__ == "__main__":
    main()
