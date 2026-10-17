"""CLI subcommands: train + enjoy (watch the walker)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gymnasium as gym
import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten

from .ppo import PPOConfig, PPOTrainer

logger = logging.getLogger(__name__)


def _configure_logging(log_file: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode="a")],
    )


def train_cmd(args) -> None:
    mx.random.seed(args.seed)

    run_dir = Path(args.run_dir)
    env_path = "BipedalWalker-v3"
    envs = [
        gym.make(env_path, hardcore=args.hardcore)
        for _ in range(args.num_envs)
    ]
    for env in envs:
        env.action_space.seed(args.seed)

    cfg = PPOConfig(
        seed=args.seed,
        total_steps=args.total_steps,
        num_steps=args.num_steps,
        num_envs=args.num_envs,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        lr=args.lr,
        anneal_lr=not args.no_anneal_lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        scale_reward=not args.no_reward_scale,
        save_every=args.save_every,
        run_dir=str(run_dir),
        hidden=tuple(args.hidden),
    )
    trainer = PPOTrainer(cfg, envs)
    _configure_logging(run_dir / "log.txt")
    if args.resume:
        trainer.load(Path(args.resume))
        logger.info("resumed policy weights from %s", args.resume)
    logger.info("config: %s", cfg)

    try:
        trainer.learn()
    finally:
        for env in envs:
            env.close()


def enjoy_cmd(args) -> None:
    from .networks import ActorCritic

    ckpt = Path(args.checkpoint)
    npz = np.load(ckpt)

    env = gym.make("BipedalWalker-v3", hardcore=args.hardcore,
                   render_mode="human" if args.render else None)
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0],
                        tuple(args.hidden))
    params = {k[2:]: mx.array(npz[k]) for k in npz.files if k.startswith("p.")}
    model.update(tree_unflatten(params))   # type: ignore[arg-type]
    mx.eval(model.parameters())

    norm_mean, norm_var = npz["norm.mean"], npz["norm.var"]

    print(f"loaded {ckpt.name} (step {int(npz['meta.steps'])}, best {float(npz['meta.best']):.1f})")
    returns = []
    for ep in range(args.episodes):
        obs, _info = env.reset(seed=args.seed + ep)
        ep_ret, done, steps = 0.0, False, 0
        while not done:
            o = ((obs - norm_mean) / np.sqrt(norm_var + 1e-8)).astype(np.float32)
            mu, _, _ = model.forward(mx.array(o[None]))
            action = np.clip(np.array(mx.clip(mu, -1.0, 1.0))[0], -1, 1).astype(np.float32)
            obs, r, term, trunc, _ = env.step(action)
            ep_ret += float(r)
            done = term or trunc
            steps += 1
        returns.append(ep_ret)
        print(f"episode {ep + 1}: return {ep_ret:8.1f}   steps {steps}")
    if returns:
        print(f"mean return over {len(returns)} episodes: {sum(returns) / len(returns):.1f}")
    env.close()


def main() -> None:   # referenced in pyproject [project.scripts]

    p = argparse.ArgumentParser(prog="mlx-walking-rl",
                                description="PPO walker trained with MLX on BipedalWalker-v3")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train a PPO walker")
    t.add_argument("--run-dir", default="runs/exp")
    t.add_argument("--total-steps", type=int, default=1_000_000)
    t.add_argument("--num-steps", type=int, default=2048)
    t.add_argument("--num-envs", type=int, default=4)
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--minibatch-size", type=int, default=512)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--no-anneal-lr", action="store_true")
    t.add_argument("--gamma", type=float, default=0.99)
    t.add_argument("--gae-lambda", type=float, default=0.95)
    t.add_argument("--clip-coef", type=float, default=0.2)
    t.add_argument("--ent-coef", type=float, default=1e-3)
    t.add_argument("--vf-coef", type=float, default=0.5)
    t.add_argument("--max-grad-norm", type=float, default=0.5)
    t.add_argument("--no-reward-scale", action="store_true")
    t.add_argument("--target-kl", type=float, default=None)
    t.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    t.add_argument("--save-every", type=int, default=10)
    t.add_argument("--hardcore", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--resume", type=str, default=None)
    t.set_defaults(func=train_cmd)

    e = sub.add_parser("enjoy", help="watch a trained policy (optionally with --render)")
    e.add_argument("--checkpoint", type=str, required=True)
    e.add_argument("--episodes", type=int, default=3)
    e.add_argument("--render", action="store_true")
    e.add_argument("--hardcore", action="store_true")
    e.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    e.add_argument("--seed", type=int, default=1)
    e.set_defaults(func=enjoy_cmd)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
