from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.ppo_agent import make_ppo
from env.candy_env import CandyEnv
from utils.seed import set_global_seed
from utils.tensorboard import make_run_dir


class RichRewardCallback:
    """Tracks per-episode metrics and logs them to TensorBoard."""

    def __init__(self) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.episode_rewards: list[float] = []
                self.episode_lengths: list[int] = []
                self.episode_invalid_rates: list[float] = []
                self.episode_cascade_counts: list[int] = []
                self.episode_cascade_rewards: list[float] = []

                self._ep_reward = 0.0
                self._ep_steps = 0
                self._ep_invalid = 0
                self._ep_cascades = 0
                self._ep_cascade_reward = 0.0

            def _on_step(self) -> bool:
                rewards = self.locals.get("rewards", [])
                dones = self.locals.get("dones", [])
                infos = self.locals.get("infos", [{}])

                for idx, (reward, done, info) in enumerate(zip(rewards, dones, infos)):
                    self._ep_reward += float(reward)
                    self._ep_steps += 1
                    if info.get("invalid", False):
                        self._ep_invalid += 1
                    if float(reward) > 0:
                        self._ep_cascades += 1
                        self._ep_cascade_reward += float(reward)

                    if done:
                        ep_len = self._ep_steps
                        invalid_rate = self._ep_invalid / max(ep_len, 1)
                        mean_cascade = (
                            self._ep_cascade_reward / self._ep_cascades
                            if self._ep_cascades > 0 else 0.0
                        )

                        self.episode_rewards.append(self._ep_reward)
                        self.episode_lengths.append(ep_len)
                        self.episode_invalid_rates.append(invalid_rate)
                        self.episode_cascade_counts.append(self._ep_cascades)
                        self.episode_cascade_rewards.append(mean_cascade)

                        ep_idx = len(self.episode_rewards)
                        self.logger.record("rollout/episode_reward", self._ep_reward)
                        self.logger.record("rollout/episode_reward_custom", self._ep_reward)
                        self.logger.record("charts/episode_reward", self._ep_reward)
                        self.logger.record("rollout/episode_length", ep_len)
                        self.logger.record("rollout/invalid_action_rate", invalid_rate)
                        self.logger.record("rollout/cascades_per_episode", self._ep_cascades)
                        self.logger.record("rollout/cascade_reward_mean", mean_cascade)

                        self._ep_reward = 0.0
                        self._ep_steps = 0
                        self._ep_invalid = 0
                        self._ep_cascades = 0
                        self._ep_cascade_reward = 0.0

                return True

        self.callback = _Callback()


def train(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    env = CandyEnv(max_moves=args.max_moves)
    tb_run_dir = make_run_dir(ROOT / args.log_dir, "ppo")
    from stable_baselines3.common.logger import configure

    model = make_ppo(
        env,
        gamma=args.gamma,
        use_maskable=not args.no_maskable,
        seed=args.seed,
    )
    model.set_logger(configure(str(tb_run_dir), ["stdout", "tensorboard"]))

    cb = RichRewardCallback()
    model.learn(total_timesteps=args.timesteps, callback=cb.callback)

    model_path = ROOT / args.model_path
    log_path = ROOT / args.log_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    with log_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["episode", "reward", "invalid_rate", "cascades", "cascade_reward_mean"],
        )
        w.writeheader()
        for idx, (r, inv, cas, cmean) in enumerate(
            zip(
                cb.callback.episode_rewards,
                cb.callback.episode_invalid_rates,
                cb.callback.episode_cascade_counts,
                cb.callback.episode_cascade_rewards,
            ),
            start=1,
        ):
            w.writerow({
                "episode": idx,
                "reward": r,
                "invalid_rate": inv,
                "cascades": cas,
                "cascade_reward_mean": cmean,
            })

    print(f"Saved PPO model to {model_path}")
    print(f"Saved training log to {log_path}")
    print(f"Saved TensorBoard logs to {tb_run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--max-moves", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--no-maskable", action="store_true")
    parser.add_argument("--model-path", type=str, default="models/ppo")
    parser.add_argument("--log-path", type=str, default="logs/ppo_rewards.csv")
    parser.add_argument("--log_dir", type=str, default="logs/tensorboard")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
