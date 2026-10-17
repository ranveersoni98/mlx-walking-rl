---
library_name: mlx
pipeline_tag: reinforcement-learning
tags:
  - reinforcement-learning
  - ppo
  - gymnasium
  - box2d
  - bipedal-walker-v3
  - apple-silicon
  - continuous-control
---

# BipedalWalker-v3 · PPO trained from scratch with MLX on the Apple GPU

A PPO agent that learned to walk in **BipedalWalker-v3** (Box2D/gymnasium).
Trained entirely with [MLX](https://github.com/ml-explore/mlx) — no PyTorch, no JAX —
on an Apple-silicon GPU in **~6 minutes** (1M env steps, ~3000 steps/s).

| Eval (deterministic) | Result |
|---|---|
| 3 episodes, seeds 42000–42002 | **281.8 / 287.8 / 287.8** |
| Mean return | **~286** |
| Episode length | ~1300–1400 of 1600 steps |

Random policy baseline: ≈ −100 (instant faceplant).

## Files

| File | What |
|---|---|
| `walker.npz` | **best** checkpoint (eval ≈ +286): policy weights, obs/reward normalizers, metadata |
| `walker_final.npz` | final-state checkpoint at step 1M (same quality within noise, seeds differ) |
| `inference.py` | standalone play script (numpy + mlx + gymnasium only) |
| `run.json` | training hyperparameters |

## Try it

```sh
pip install mlx gymnasium box2d-py numpy
python inference.py walker.npz --episodes 3            # headless
python inference.py walker.npz --episodes 1 --render   # watch it walk
```

## Recipe (CleanRL-style PPO)

`total_steps=1M, num_envs=4, steps_per_iter=2048, epochs=10, minibatch=512,
lr=3e-4 linear-anneal, gamma=0.99, gae_lambda=0.95, clip=0.2, ent_coef=1e-3,
vf_coef=0.5, max_grad_norm=0.5`, 64x64 tanh actor/critic heads.

Stabilizers that made it work on this env (each one was a real failure mode):

1. **GAE keeps the `−V(s_t)` TD-error term.** Omitting it silently turns the
   advantage recursion into a return-accumulator whose critic echo grows by
   ~1/(1−γλ) and explodes the value loss (109 → 455,792) while freezing the policy.
2. **Clamped normalization** — obs and rewards are running-normalized and
   clamped to ±10 (SB3 `VecNormalize`-style); without this, near-constant
   observation dims hit ±10³ targets after Welford warm-up.
3. **No value-clip** — with non-stationary targets, SB3-style value clipping
   confines the critic to stale values and starves the policy head of the
   global gradient-norm budget.
4. **Raw actions in the buffer** (clip only what reaches the env), so old
   log-probs are recomputed consistently.
5. Milestone: the walker first learned the *stand-still* local optimum
   (survives 1600 steps, −120 return) and escaped once the critic stabilized.

Source: https://github.com/ranveersoni/mlx-walking-rl
