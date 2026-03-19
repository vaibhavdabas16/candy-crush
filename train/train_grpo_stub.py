from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.llm_agent import MockLLMAgent
from env.candy_env import CandyEnv
from utils.seed import set_global_seed
from utils.state_to_text import state_to_text


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    env = CandyEnv(max_moves=args.max_moves)
    agent = MockLLMAgent()
    log_path = ROOT / args.log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, int | float | str]] = []

    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        step_idx = 0
        print(f"\nEpisode {episode}")

        while not done:
            step_idx += 1
            candidates = agent.propose_actions(env)
            ranked = agent.rank_actions(env, candidates)
            action = ranked[0][1]
            print(state_to_text(env, max_actions=8))
            print(f"Candidates: {ranked}")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rows.append(
                {
                    "episode": episode,
                    "step": step_idx,
                    "chosen_action": action,
                    "reward": reward,
                    "candidates": repr(ranked),
                    "score": info["score"],
                }
            )

    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode", "step", "chosen_action", "reward", "candidates", "score"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved GRPO stub log to {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-moves", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-path", type=str, default="logs/grpo_stub.csv")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
