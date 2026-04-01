from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from env.candy_env import CandyEnv
from gui.viewer import CandyViewer, load_policy


def repo_path_or_id(value: str) -> str | Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    if value.count("/") == 1 and not value.startswith("."):
        return value
    return ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        choices=["manual", "random", "greedy", "dqn", "ppo", "grpo", "llm_grpo"],
        default="manual",
    )
    parser.add_argument("--max-moves", type=int, default=20)
    parser.add_argument("--dqn-path",  type=str, default="models/dqn.pt")
    parser.add_argument("--ppo-path",  type=str, default="models/ppo")
    parser.add_argument("--grpo-path", type=str, default="models/grpo_candy.pt")
    parser.add_argument("--llm-grpo-path", type=str, default="models/llm_grpo_candy/qwen35_9b/final_plus30")
    parser.add_argument("--llm-model-name", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--agent-delay", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = CandyEnv(max_moves=args.max_moves)
    policy = load_policy(
        args.agent,
        ROOT / args.dqn_path,
        ROOT / args.ppo_path,
        env,
        ROOT / args.grpo_path,
        repo_path_or_id(args.llm_grpo_path),
        args.llm_model_name,
    )
    viewer = CandyViewer(env, policy=policy, mode=args.agent)
    viewer.config.agent_delay = args.agent_delay
    viewer.run()


if __name__ == "__main__":
    main()
