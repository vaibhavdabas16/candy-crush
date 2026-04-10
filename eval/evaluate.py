from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from agents.baselines import GreedyPolicy, RandomPolicy
from agents.saved_models import load_saved_policy, saved_policy_exists
from env.candy_env import CandyEnv
from utils.metrics import summarize_scores
from utils.seed import set_global_seed


def run_episode(env: CandyEnv, policy, seed: int) -> tuple[float, int, int]:
    """Returns (total_reward, invalid_steps, cascade_count)."""
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    invalid_steps = 0
    cascades = 0

    while not done:
        module = policy.__class__.__module__
        if module.startswith("stable_baselines3") or module.startswith("sb3_contrib"):
            try:
                action, _ = policy.predict(obs, deterministic=True, action_masks=env.action_masks())
            except TypeError:
                action, _ = policy.predict(obs, deterministic=True)
        else:
            action, _ = policy.predict(obs, env=env, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        if info.get("invalid", False):
            invalid_steps += 1
        if reward > 0:
            cascades += 1
        done = terminated or truncated

    return float(total_reward), invalid_steps, cascades


def evaluate_policy(
    name: str,
    policy,
    args: argparse.Namespace,
    writer,
) -> list[float]:
    scores: list[float] = []
    invalid_rates: list[float] = []
    cascade_counts: list[int] = []

    for seed in range(args.seed, args.seed + args.seeds):
        set_global_seed(seed)
        env = CandyEnv(max_moves=args.max_moves)
        for episode in range(args.episodes):
            score, invalids, cascades = run_episode(env, policy, seed * 10_000 + episode)
            scores.append(score)
            invalid_rates.append(invalids / args.max_moves)
            cascade_counts.append(cascades)

    metrics = summarize_scores(name, scores, args.max_moves)
    avg_invalid = float(np.mean(invalid_rates))
    avg_cascades = float(np.mean(cascade_counts))

    print(
        f"{metrics.policy:10s} | avg={metrics.average_score:8.2f} "
        f"| per_move={metrics.score_per_move:7.2f} "
        f"| var={metrics.variance:10.2f} "
        f"| invalid={avg_invalid:.3f} "
        f"| cascades={avg_cascades:.1f} "
        f"| episodes={metrics.episodes}"
    )

    if writer is not None:
        tag = name.lower()
        writer.add_scalar(f"eval/{tag}/avg_score", metrics.average_score, 0)
        writer.add_scalar(f"eval/{tag}/score_per_move", metrics.score_per_move, 0)
        writer.add_scalar(f"eval/{tag}/score_variance", metrics.variance, 0)
        writer.add_scalar(f"eval/{tag}/invalid_action_rate", avg_invalid, 0)
        writer.add_scalar(f"eval/{tag}/cascades_per_episode", avg_cascades, 0)

        # Score distribution as histogram
        writer.add_histogram(f"eval/{tag}/score_distribution", np.array(scores), 0)

        # Per-episode curve so you can see consistency
        for i, s in enumerate(scores):
            writer.add_scalar(f"eval/{tag}/episode_score", s, i)

    return scores


def main(args: argparse.Namespace) -> None:
    from torch.utils.tensorboard import SummaryWriter
    from utils.tensorboard import make_run_dir

    tb_run_dir = make_run_dir(ROOT / args.log_dir, "eval")
    writer = SummaryWriter(log_dir=str(tb_run_dir))

    print("Policy     | Average score | Score/move | Variance   | Invalid | Cascades | Episodes")
    print("-" * 90)

    all_scores: dict[str, list[float]] = {}

    all_scores["random"] = evaluate_policy("random", RandomPolicy(), args, writer)
    all_scores["greedy"] = evaluate_policy("greedy", GreedyPolicy(), args, writer)

    if saved_policy_exists("dqn", dqn_path=args.dqn_path):
        dqn = load_saved_policy("dqn", dqn_path=args.dqn_path)
        all_scores["dqn"] = evaluate_policy("dqn", dqn, args, writer)
    else:
        print(f"dqn        | skipped, model not found at {ROOT / args.dqn_path}")

    if saved_policy_exists("ppo", ppo_path=args.ppo_path):
        env = CandyEnv(max_moves=args.max_moves)
        ppo = load_saved_policy("ppo", ppo_path=args.ppo_path, env=env)
        all_scores["ppo"] = evaluate_policy("ppo", ppo, args, writer)
    else:
        print(f"ppo        | skipped, model not found at {ROOT / args.ppo_path}.zip")

    # Side-by-side bar chart of average scores
    if len(all_scores) > 1:
        for name, scores in all_scores.items():
            writer.add_scalar("eval/comparison/avg_score", float(np.mean(scores)), list(all_scores.keys()).index(name))

        # Overlay score distributions for all policies on one histogram
        for name, scores in all_scores.items():
            writer.add_histogram("eval/comparison/score_distributions", np.array(scores), list(all_scores.keys()).index(name))

    writer.flush()
    writer.close()
    print(f"\nTensorBoard eval logs saved to {tb_run_dir}")
    print("Run: tensorboard --logdir logs/tensorboard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-moves", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--dqn-path", type=str, default="models/dqn.pt")
    parser.add_argument("--ppo-path", type=str, default="models/ppo")
    parser.add_argument("--log_dir", type=str, default="logs/tensorboard")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
