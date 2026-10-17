"""PPO trainer for continuous control on the Apple GPU with MLX."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.optimizers import Adam
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from .networks import ActorCritic
from .running_norm import RunningMeanStd


@dataclass
class PPOConfig:
    seed: int = 0
    total_steps: int = 1_000_000
    num_steps: int = 2048        # rollout length per env, per iteration
    num_envs: int = 4            # parallel envs stepping together
    epochs: int = 10             # optimization epochs per rollout
    minibatch_size: int = 512
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    use_value_clip: bool = False  # SB3-style value clipping; needs slow targets
    scale_reward: bool = True     # divide rewards by std of running discounted return
    clip_reward: float = 10.0     # clamp scaled rewards, SB3 VecNormalize-style
    hidden: tuple[int, ...] = field(default=(64, 64))
    save_every: int = 10         # checkpoint every N iterations
    run_dir: str = "runs/exp"


class PPOTrainer:
    def __init__(self, cfg: PPOConfig, envs: list) -> None:
        self.cfg = cfg
        self.envs = envs
        self.n_envs = len(envs)
        assert self.n_envs >= 1, "need at least one environment"

        obs_dim = envs[0].observation_space.shape[0]
        act_dim = envs[0].action_space.shape[0]
        self.obs_dim, self.act_dim = obs_dim, act_dim

        self.model = ActorCritic(obs_dim, act_dim, tuple(cfg.hidden))
        mx.eval(self.model.parameters())

        self.optimizer = Adam(learning_rate=cfg.lr)
        self.rms = RunningMeanStd((obs_dim,), clip=10.0)
        self.rew_rms = RunningMeanStd((1,))

        self._loss_and_grad = nn.value_and_grad(self.model, self._loss_fn)
        self._diag: dict[str, mx.array] = {}

        self.rng = np.random.default_rng(cfg.seed)
        self.best_reward = -np.inf

        run_dir = Path(cfg.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        with (run_dir / "run.json").open("w") as f:
            json.dump({
                **asdict(cfg),
                "hidden": list(cfg.hidden),
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "env": envs[0].spec.id if envs[0].spec else "?",
            }, f, indent=2)

    # ------------------------------------------------------------------ #
    # acting / value estimation                                           #
    # ------------------------------------------------------------------ #
    def _forward_np(self, obs_norm: np.ndarray, deterministic: bool = False):
        """Run the policy on normalized obs [E, obs_dim]; evaluate eagerly.

        Returns (env_actions [E, A] clipped to [-1, 1], raw sampled actions [E, A],
                 log_probs [E] of the raw actions, values [E]).
        """
        x = mx.array(obs_norm.astype(np.float32))
        mu, log_std, value = self.model.forward(x)
        if deterministic:
            act = mu
        else:
            std = mx.exp(log_std)
            act = mu + std * mx.random.normal(mu.shape)
        logp = self.model.log_prob(mu, log_std, act)
        mx.eval(act, logp, mu, value)
        raw = np.array(act, dtype=np.float32)
        env_act = np.clip(raw, -1.0, 1.0)
        return env_act, raw, np.array(logp, dtype=np.float32), np.array(value, dtype=np.float32)

    def _values_np(self, obs_norm: np.ndarray) -> np.ndarray:
        x = mx.array(obs_norm.astype(np.float32))
        _, _, value = self.model.forward(x)
        mx.eval(value)
        return np.array(value, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # rollout collection                                                  #
    # ------------------------------------------------------------------ #
    def collect_rollout(self) -> dict[str, np.ndarray]:
        """Collect one PPO rollout across all parallel envs.

        Bootstrapping is handled correctly:
          * termination (robot fell):      bootstrap value = 0
          * truncation (time limit hit):   bootstrap = V(terminal obs)

        Returns flat numpy arrays for the update; episode stats accumulate
        in trainer state.
        """
        cfg, T, E = self.cfg, self.cfg.num_steps, self.n_envs
        obs_dim, act_dim = self.obs_dim, self.act_dim

        obs_buf = np.zeros((T, E, obs_dim), dtype=np.float32)
        act_buf = np.zeros((T, E, act_dim), dtype=np.float32)
        logp_buf = np.zeros((T, E), dtype=np.float32)
        val_buf = np.zeros((T, E), dtype=np.float32)
        rew_buf = np.zeros((T, E), dtype=np.float32)
        done_buf = np.zeros((T, E), dtype=np.float32)   # term or trunc
        # value bootstrap for envs whose episode ENDED at t (else NaN -> unused)
        nextval_override = np.zeros((T, E), dtype=np.float32) + np.nan

        for t in range(T):
            self.rms.update(self._obs_raw)                       # raw obs -> normalizer
            obs_t = self.rms.normalize(self._obs_raw).astype(np.float32)
            obs_buf[t] = obs_t
            env_actions, raw_actions, logp, values = self._forward_np(obs_t)

            for i, env in enumerate(self.envs):
                n_raw, rew_raw, terminated, truncated, _info = env.step(env_actions[i])
                rew_raw = float(rew_raw)
                done = terminated or truncated

                # reward scaling: keep value targets O(1) so the critic (Adam,
                # small lr) can actually track them; report raw returns for logging
                if self.cfg.scale_reward:
                    self._ret_est[i] = cfg.gamma * self._ret_est[i] + rew_raw
                    rew = rew_raw / np.sqrt(self.rew_rms.var[0] + 1e-8)
                    if self.cfg.clip_reward > 0:
                        rew = float(np.clip(rew, -self.cfg.clip_reward, self.cfg.clip_reward))
                else:
                    rew = rew_raw
                self.rew_rms.update(self._ret_est)

                rew_buf[t, i] = rew
                done_buf[t, i] = float(done)
                self._obs_raw[i] = n_raw

                if done:
                    if truncated:
                        # V of the true terminal state (before reset)
                        term_n = self.rms.normalize(self._obs_raw[i][None])
                        nextval_override[t, i] = self._values_np(term_n)[0]
                    else:
                        nextval_override[t, i] = 0.0
                    self._episode_returns.append(self._ep_return[i] + rew_raw)
                    self._episode_lengths.append(self._ep_len[i] + 1)
                    self._ep_return[i] = 0.0
                    self._ep_len[i] = 0
                    n_raw, _ = env.reset()   # seed only on first reset per env
                    self._obs_raw[i] = n_raw
                else:
                    self._ep_return[i] += rew_raw
                    self._ep_len[i] += 1

            act_buf[t] = raw_actions  # unclipped: log-probs are recomputed on these
            logp_buf[t] = logp
            val_buf[t] = values

        final_nextval = self._values_np(self.rms.normalize(self._obs_raw)).astype(np.float32)

        # ---- GAE, batched over envs (CleanRL-style) ----
        adv_buf = np.zeros((T, E), dtype=np.float32)
        lastgaelam = np.zeros(E, dtype=np.float32)
        for t in range(T - 1, -1, -1):
            if t == T - 1:
                nextvals = final_nextval
            else:
                nextvals = val_buf[t + 1]
            # bootstrap: 0 at termination, V(terminal obs) at truncation
            nextvals = np.where(done_buf[t] > 0, nextval_override[t], nextvals)
            # TD error: r + gamma*V(s') - V(s)    (the -V(s) is critical!)
            delta = rew_buf[t] + cfg.gamma * nextvals - val_buf[t]
            lastgaelam = delta + cfg.gamma * cfg.gae_lambda * (1.0 - done_buf[t]) * lastgaelam
            adv_buf[t] = lastgaelam

        ret_buf = adv_buf + val_buf

        return {
            "obs": obs_buf.reshape(T * E, obs_dim),
            "act": act_buf.reshape(T * E, act_dim),
            "adv": adv_buf.reshape(T * E),
            "ret": ret_buf.reshape(T * E),
            "old_values": val_buf.reshape(T * E),
            "old_logp": logp_buf.reshape(T * E),
        }

    # ------------------------------------------------------------------ #
    # optimization                                                        #
    # ------------------------------------------------------------------ #
    def _loss_fn(self, obs, act, adv, ret, old_values, old_logp):
        mu, log_std, value = self.model.forward(obs)
        logstd_b = mx.broadcast_to(log_std, mu.shape)
        logp = self.model.log_prob(mu, logstd_b, act)
        entropy = self.model.entropy(logstd_b)

        ratio = mx.exp(logp - old_logp)
        adv_norm = (adv - adv.mean()) / (mx.sqrt(mx.var(adv)) + 1e-8)
        pg1 = ratio * adv_norm
        pg2 = mx.clip(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef) * adv_norm
        pg_loss = -mx.minimum(pg1, pg2).mean()

        if self.cfg.use_value_clip:
            v_clip = old_values + mx.clip(value - old_values,
                                          -self.cfg.clip_coef, self.cfg.clip_coef)
            v_loss_un = mx.square(value - ret)
            v_loss_cl = mx.square(v_clip - ret)
            vf_loss = 0.5 * mx.maximum(v_loss_un, v_loss_cl).mean()
        else:
            vf_loss = 0.5 * mx.square(value - ret).mean()

        loss = pg_loss + self.cfg.vf_coef * vf_loss - self.cfg.ent_coef * entropy

        self._diag = {
            "pi_loss": mx.stop_gradient(pg_loss),
            "vf_loss": mx.stop_gradient(vf_loss),
            "entropy": mx.stop_gradient(entropy),
            "approx_kl": mx.stop_gradient(mx.mean(old_logp - logp)),
            "clipfrac": mx.stop_gradient(
                mx.mean((mx.abs(ratio - 1.0) > self.cfg.clip_coef).astype(mx.float32))),
        }
        return loss

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        n = batch["obs"].shape[0]
        B = min(self.cfg.minibatch_size, n)

        metrics: dict[str, list[float]] = {}
        stop_epoch = False
        for _ in range(self.cfg.epochs):
            perm = self.rng.permutation(n)
            for start in range(0, n, B):
                idx = perm[start:start + B]
                mb = {k: mx.array(v[idx]) for k, v in batch.items()}
                loss, grads = self._loss_and_grad(**mb)

                grads = self._clip_grads(grads, self.cfg.max_grad_norm)
                self.optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), self.optimizer.state)

                loss_f = float(loss)
                metrics.setdefault("loss", []).append(loss_f)
                for k, d in self._diag.items():
                    metrics.setdefault(k, []).append(float(d))
                if self.cfg.target_kl is not None \
                        and metrics["approx_kl"][-1] > self.cfg.target_kl:
                    stop_epoch = True
                    break
            if stop_epoch:
                break

        return {
            k: float(np.mean(v)) for k, v in metrics.items()
        } | {"stopped_early": stop_epoch}

    @staticmethod
    def _clip_grads(grads, max_norm: float):
        norm = mx.sqrt(sum(mx.square(g).sum() for _, g in tree_flatten(grads)))
        scale = mx.minimum(1.0, max_norm / (norm + 1e-6))
        return tree_map(lambda g: g * scale, grads)

    # ------------------------------------------------------------------ #
    # main loop                                                           #
    # ------------------------------------------------------------------ #
    def learn(self) -> None:
        cfg = self.cfg
        samples_per_iter = cfg.num_steps * self.n_envs
        n_iters = max(1, cfg.total_steps // samples_per_iter)

        self.reset()
        for it in range(1, n_iters + 1):
            t0 = time.time()
            batch = self.collect_rollout()
            stats = self.update(batch)
            dt = time.time() - t0

            if cfg.anneal_lr:
                frac = 1.0 - (it - 1.0) / n_iters
                self.optimizer.learning_rate = cfg.lr * max(frac, 0.0)

            global_step = it * samples_per_iter
            sps = int(samples_per_iter / dt) if dt > 0 else 0
            ep_mean = (float(np.mean(self._episode_returns[-100:]))
                       if self._episode_returns else float("nan"))
            line = (f"it {it:>4}/{n_iters} | step {global_step:>8} | sps {sps:>6} | "
                    f"reward100 {ep_mean:>8.1f} | best {self.best_reward:>8.1f} | "
                    f"pi {stats['pi_loss']:>8.4f} | vf {stats['vf_loss']:>9.3f} | "
                    f"ent {stats['entropy']:>6.3f} | kl {stats['approx_kl']:>7.5f} | "
                    f"clipf {stats['clipfrac']:.3f} | lr {self.optimizer.learning_rate:.2e}")
            print(line, flush=True)

            if (it % cfg.save_every == 0) or it == n_iters:
                self.save(self.run_dir / f"ckpt_iter{it:05d}.npz", global_step=global_step)
                self.save(self.run_dir / "ckpt_latest.npz", global_step=global_step)
            if ep_mean > self.best_reward:
                self.best_reward = ep_mean
                self.save(self.run_dir / "ckpt_best.npz", global_step=global_step)

    def reset(self) -> None:
        """Reset envs, episode stats and normalizer inputs."""
        self._obs_raw = np.zeros((self.n_envs, self.obs_dim), dtype=np.float32)
        self._ep_return = np.zeros(self.n_envs, dtype=np.float64)
        self._ep_len = np.zeros(self.n_envs, dtype=np.int64)
        self._ret_est = np.zeros(self.n_envs, dtype=np.float64)  # discounted-return EMA
        self._episode_returns: list[float] = []
        self._episode_lengths: list[int] = []
        for i, env in enumerate(self.envs):
            self._obs_raw[i], _ = env.reset(seed=self.cfg.seed + i)

    # ------------------------------------------------------------------ #
    # checkpointing                                                       #
    # ------------------------------------------------------------------ #
    def save(self, path: Path, global_step: int = 0) -> None:
        arrays = {f"p.{k}": np.array(v) for k, v in tree_flatten(self.model.parameters())}
        arrays["norm.mean"] = self.rms.mean
        arrays["norm.var"] = self.rms.var
        arrays["norm.count"] = np.array(self.rms.count)
        arrays["norm.rew_mean"] = self.rew_rms.mean
        arrays["norm.rew_var"] = self.rew_rms.var
        arrays["norm.rew_count"] = np.array(self.rew_rms.count)
        arrays["meta.steps"] = np.array(global_step, dtype=np.int64)
        arrays["meta.best"] = np.array(self.best_reward, dtype=np.float64)
        arrays["meta.lr"] = np.array(float(self.optimizer.learning_rate), dtype=np.float64)
        np.savez(path, **arrays)

    def load(self, path: Path) -> None:
        z = np.load(path)
        params = {k[2:]: mx.array(z[k]) for k in z.files if k.startswith("p.")}
        self.model.update(tree_unflatten(params))
        mx.eval(self.model.parameters())
        self.rms.mean = z["norm.mean"].copy()
        self.rms.var = z["norm.var"].copy()
        self.rms.count = float(z["norm.count"])
        if "norm.rew_var" in z.files:
            self.rew_rms.mean = z["norm.rew_mean"].copy()
            self.rew_rms.var = z["norm.rew_var"].copy()
            self.rew_rms.count = float(z["norm.rew_count"])
        if "meta.best" in z.files:
            self.best_reward = float(z["meta.best"])
