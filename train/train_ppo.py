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


class RewardLoggerCallback:
    def __init__(self) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.episode_rewards: list[float] = []
                self.current_rewards: list[float] = []

            def _on_step(self) -> bool:
                rewards = self.locals.get("rewards", [])
                dones = self.locals.get("dones", [])
                for idx, reward in enumerate(rewards):
                    if idx >= len(self.current_rewards):
                        self.current_rewards.append(0.0)
                    self.current_rewards[idx] += float(reward)
                    if dones[idx]:
                        self.episode_rewards.append(self.current_rewards[idx])
                        self.current_rewards[idx] = 0.0
                return True

        self.callback = _Callback()


def train(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    env = CandyEnv(max_moves=args.max_moves)
    model = make_ppo(env, gamma=args.gamma, use_maskable=not args.no_maskable, seed=args.seed)

    reward_logger = RewardLoggerCallback()
    model.learn(total_timesteps=args.timesteps, callback=reward_logger.callback)

    model_path = ROOT / args.model_path
    log_path = ROOT / args.log_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward"])
        writer.writeheader()
        for idx, reward in enumerate(reward_logger.callback.episode_rewards, start=1):
            writer.writerow({"episode": idx, "reward": reward})

    print(f"Saved PPO model to {model_path}")
    print(f"Saved training log to {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--max-moves", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--no-maskable", action="store_true")
    parser.add_argument("--model-path", type=str, default="models/ppo")
    parser.add_argument("--log-path", type=str, default="logs/ppo_rewards.csv")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
