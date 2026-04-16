from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.dqn_agent import DQNAgent, DQNConfig
from env.candy_env import CandyEnv
from utils.seed import set_global_seed


def train(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    env = CandyEnv(max_moves=args.max_moves)

    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=DQNConfig(gamma=args.gamma, lr=args.lr, batch_size=args.batch_size),
    )

    rewards_log: list[dict[str, float | int]] = []
    global_step = 0
    for episode in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + episode)
        done = False
        episode_reward = 0.0
        losses: list[float] = []

        while not done:
            action = agent.select_action(obs, info["valid_action_mask"], training=True)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            agent.store(
                obs,
                action,
                reward,
                next_obs,
                done,
                next_info["valid_action_mask"],
            )
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

            obs = next_obs
            info = next_info
            episode_reward += reward
            global_step += 1

        row = {
            "episode": episode,
            "reward": episode_reward,
            "epsilon": agent.epsilon(),
            "loss": sum(losses) / len(losses) if losses else 0.0,
            "steps": global_step,
        }
        rewards_log.append(row)
        if episode % args.log_every == 0 or episode == 1:
            print(
                f"episode={episode} reward={episode_reward:.1f} "
                f"epsilon={row['epsilon']:.3f} loss={row['loss']:.4f}"
            )

    model_path = ROOT / args.model_path
    log_path = ROOT / args.log_path
    agent.save(model_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "epsilon", "loss", "steps"])
        writer.writeheader()
        writer.writerows(rewards_log)

    print(f"Saved DQN model to {model_path}")
    print(f"Saved training log to {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-moves", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--model-path", type=str, default="models/dqn.pt")
    parser.add_argument("--log-path", type=str, default="logs/dqn_rewards.csv")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
