# mlx-walking-rl

A PPO agent that teaches itself to walk in **BipedalWalker-v3** — implemented
entirely in [MLX](https://github.com/ml-explore/mlx) and trained on the Apple
GPU. No PyTorch, no JAX.

| Policy | Mean return | Avg. episode |
| --- | --- | --- |
| Random init | ≈ −100 | falls in ~20 steps |
| PPO, 1M steps | **≈ +286** | survives ~1,350 of 1,600 steps |

Deterministic eval: 281.8 / 287.8 / 287.8. Training took about
six minutes at ~3,000 env steps/s on a laptop GPU. Trained weights and a
standalone play script live on the Hub:
[`RanveerSoni98/bipedal-walker-ppo-mlx`](https://huggingface.co/RanveerSoni98/bipedal-walker-ppo-mlx).

## Install

```sh
git clone https://github.com/ranveersoni98/mlx-walking-rl
cd mlx-walking-rl
uv sync          # or: pip install -e .
```

Requires Python 3.12+ and an Apple-silicon Mac (MLX targets the Metal GPU).

## Train

```sh
uv run mlx-walking-rl train --run-dir runs/walker1
```

Defaults train 1M environment steps (4 parallel envs). Progress streams to the
terminal and to `runs/walker1/log.txt`; checkpoints are written every 10
iterations:

```sh
runs/walker1/ckpt_latest.npz   # most recent
runs/walker1/ckpt_best.npz     # best running eval
runs/walker1/ckpt_iter00010.npz
```

Resume from a checkpoint (weights carry over, optimizer restarts):

```sh
uv run mlx-walking-rl train --run-dir runs/walker2 --resume runs/walker1/ckpt_best.npz
```

## Watch it walk

```sh
uv run mlx-walking-rl enjoy --checkpoint runs/walker1/ckpt_best.npz --render
```

## Design

- `src/mlx_walking_rl/networks.py` — actor and critic are two 64×64 tanh MLPs;
  the Gaussian policy head has a state-independent, learned log-std.
- `src/mlx_walking_rl/ppo.py` — the PPO trainer: parallel rollouts (4 envs),
  GAE, minibatch updates with ratio clipping, entropy bonus, gradient-norm
  clipping, linear LR annealing, checkpointing.
- `src/mlx_walking_rl/running_norm.py` — Welford observation/reward
  normalization with clamping.
- `src/mlx_walking_rl/cli.py` — the `train` / `enjoy` CLI.

Hyperparameters (CleanRL-style defaults):
`num_envs=4 · steps_per_iter=2048 · epochs=10 · minibatch=512 · lr=3e-4 (linear
anneal) · gamma=0.99 · gae_lambda=0.95 · clip=0.2 · ent_coef=1e-3 · vf_coef=0.5 ·
max_grad_norm=0.5`.

## Notes for training continuous control with PPO

Four pitfalls, all hit (and fixed) while getting this env to walk — each one is
silent in logs but fatal:

1. **GAE must subtract `V(s_t)` in the TD error.** Dropping it turns the
   advantage recursion into a return-accumulator whose critic-echo inflates by
   `1/(1−γλ)`; the value loss then explodes exponentially while the policy
   freezes. `tests/test_gae.py` compares this repo's GAE against an independent
   λ-return reference to catch regressions here.
2. **Clamp everything you normalize.** Running-mean normalizers with a tiny
   initial sample count let the variance of near-constant dims (Box2D lidars)
   collapse to ~0, blowing normalized values up to ±10³. Observations are
   clamped to ±10; rewards are divided by the running std of the discounted
   return and clamped ±10 (SB3 `VecNormalize`-style).
3. **Value clipping is off by default.** With non-stationary targets across
   iterations, SB3-style value clipping confines the critic to its stale
   collection-time values; the frozen value head then eats the entire global
   gradient-norm budget and the policy head receives no signal.
4. **Store unclipped actions in the buffer.** Log-probs are recomputed on raw
   samples; only the actions sent to the env are clipped. Mixing the two makes
   the importance ratio wrong by design (visible as constant ±0.9 KL bias).

Bonus: the agent first settles into the classic *stand-still* local optimum
(survives full episodes, scores ≈ −120) and escapes once the critic
stabilizes.

## Development

```sh
uv add --dev pytest ruff      # already in dev deps
uv run pytest                 # GAE correctness test
uv run ruff check .
```

Utility scripts:

```sh
uv run python scripts/inspect_rollout.py runs/walker1/ckpt_latest.npz   # rollout + normalizer internals
uv run python scripts/trace_training.py                                  # per-iteration V/target/adv trace
```

To publish weights, stage them in `hf/` (`walker.npz`, `walker_final.npz`,
model card, inference script) and run:

```sh
hf upload RanveerSoni98/bipedal-walker-ppo-mlx hf/ . --type model
```

## License

[MIT](LICENSE)
