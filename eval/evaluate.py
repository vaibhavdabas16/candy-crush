from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.baselines import GreedyPolicy, RandomPolicy
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import load_ppo
from env.candy_env import CandyEnv
from utils.metrics import summarize_scores
from utils.seed import set_global_seed


def run_episode(env: CandyEnv, policy, seed: int) -> float:
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0

    while not done:
        module = policy.__class__.__module__
        if module.startswith("stable_baselines3") or module.startswith("sb3_contrib"):
            try:
                action, _ = policy.predict(obs, deterministic=True, action_masks=env.action_masks())
            except TypeError:
                action, _ = policy.predict(obs, deterministic=True)
        else:
            action, _ = policy.predict(obs, env=env, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += reward
        done = terminated or truncated

    return float(total_reward)


def evaluate_policy(name: str, policy, args: argparse.Namespace) -> list[float]:
    scores: list[float] = []
    for seed in range(args.seed, args.seed + args.seeds):
        set_global_seed(seed)
        env = CandyEnv(max_moves=args.max_moves)
        for episode in range(args.episodes):
            scores.append(run_episode(env, policy, seed * 10_000 + episode))

    metrics = summarize_scores(name, scores, args.max_moves)
    print(
        f"{metrics.policy:10s} | avg={metrics.average_score:8.2f} "
        f"| per_move={metrics.score_per_move:7.2f} "
        f"| var={metrics.variance:10.2f} | episodes={metrics.episodes}"
    )
    return scores


def main(args: argparse.Namespace) -> None:
    print("Policy     | Average score | Score/move | Variance | Episodes")
    print("-" * 68)
    evaluate_policy("random", RandomPolicy(), args)
    evaluate_policy("greedy", GreedyPolicy(), args)

    dqn_path = ROOT / args.dqn_path
    if dqn_path.exists():
        dqn = DQNAgent.load(dqn_path)
        evaluate_policy("dqn", dqn, args)
    else:
        print(f"dqn        | skipped, model not found at {dqn_path}")

    ppo_path = ROOT / args.ppo_path
    if ppo_path.exists() or Path(str(ppo_path) + ".zip").exists():
        env = CandyEnv(max_moves=args.max_moves)
        ppo = load_ppo(ppo_path, env=env)
        evaluate_policy("ppo", ppo, args)
    else:
        print(f"ppo        | skipped, model not found at {ppo_path}.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-moves", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--dqn-path", type=str, default="models/dqn.pt")
    parser.add_argument("--ppo-path", type=str, default="models/ppo")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
