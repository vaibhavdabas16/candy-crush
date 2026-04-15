"""Terminal demo for the four baseline agents (random, greedy, DQN, PPO).

For each seed in --seeds, every agent plays a full max_moves episode in
turn. Each step prints the ANSI board and the action taken. At the end a
comparison table shows total reward per (agent, seed) plus per-agent
averages, and the same data is written to logs/baselines_comparison.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.baselines import GreedyPolicy, RandomPolicy
from agents.saved_models import load_saved_policy, saved_policy_exists
from env.candy_env import CandyEnv

DEFAULT_AGENTS = ["random", "greedy", "dqn", "ppo"]


def predict_action(policy, obs, env: CandyEnv) -> int:
    module = policy.__class__.__module__
    if module.startswith("stable_baselines3") or module.startswith("sb3_contrib"):
        try:
            action, _ = policy.predict(obs, deterministic=True, action_masks=env.action_masks())
        except TypeError:
            action, _ = policy.predict(obs, deterministic=True)
        return int(action)
    action, _ = policy.predict(obs, env=env, deterministic=True)
    return int(action)


def play_episode(policy, agent_name: str, seed: int, max_moves: int) -> float:
    env = CandyEnv(max_moves=max_moves)
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    print(f"\n----- agent={agent_name}  seed={seed} -----")
    for step in range(1, max_moves + 1):
        print(f"\n[step {step}]")
        print(env.render(mode="ansi"))
        action = predict_action(policy, obs, env)
        pos_a, pos_b = env.decode_action(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        invalid = " INVALID" if info.get("invalid", False) else ""
        print(
            f"action={action} swap {pos_a} <-> {pos_b}  reward={reward:+.2f} "
            f"score={env.score:.2f}  moves_left={env.moves_left}{invalid}",
            flush=True,
        )
        if terminated or truncated:
            break
    print(f"\n>>> {agent_name} seed={seed} TOTAL REWARD = {total_reward:.2f}")
    return total_reward


def build_policy(name: str, args: argparse.Namespace, env: CandyEnv):
    if name == "random":
        return RandomPolicy()
    if name == "greedy":
        return GreedyPolicy()
    if name == "dqn":
        if not saved_policy_exists("dqn", dqn_path=args.dqn_path):
            print(f"  [skip] dqn: model not found at {args.dqn_path}")
            return None
        return load_saved_policy("dqn", dqn_path=args.dqn_path)
    if name == "ppo":
        if not saved_policy_exists("ppo", ppo_path=args.ppo_path):
            print(f"  [skip] ppo: model not found at {args.ppo_path}.zip")
            return None
        return load_saved_policy("ppo", ppo_path=args.ppo_path, env=env)
    raise ValueError(f"Unknown agent: {name}")


def print_table(agents_used: list[str], seeds: list[int], results: dict[str, dict[int, float]]) -> None:
    width = 78
    print("\n" + "=" * width)
    print("Comparison Table  (total reward per agent per seed)")
    print("=" * width)
    header = f"{'seed':>6s} | " + " | ".join(f"{a:>10s}" for a in agents_used)
    print(header)
    print("-" * len(header))
    for seed in seeds:
        row = f"{seed:>6d} | " + " | ".join(
            f"{results[a].get(seed, float('nan')):>10.2f}" for a in agents_used
        )
        print(row)
    print("-" * len(header))
    avg_row = f"{'avg':>6s} | "
    avg_parts = []
    for a in agents_used:
        scores = list(results[a].values())
        avg_parts.append(f"{sum(scores)/len(scores):>10.2f}" if scores else f"{'-':>10s}")
    print(avg_row + " | ".join(avg_parts))
    print("=" * width)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--agents", nargs="+", default=DEFAULT_AGENTS, choices=DEFAULT_AGENTS)
    p.add_argument("--max-moves", type=int, default=20)
    p.add_argument("--dqn-path", default="models/dqn.pt")
    p.add_argument("--ppo-path", default="models/ppo")
    p.add_argument("--log-dir", default="logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    print(
        f"Baselines demo: seeds={args.seeds}  max_moves={args.max_moves}  agents={args.agents}\n"
    )

    init_env = CandyEnv(max_moves=args.max_moves)
    policies: dict[str, object] = {}
    for name in args.agents:
        try:
            pol = build_policy(name, args, init_env)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {name}: {type(e).__name__}: {e}")
            continue
        if pol is not None:
            policies[name] = pol

    if not policies:
        print("No agents were loaded. Aborting.")
        sys.exit(1)

    agents_used = [a for a in args.agents if a in policies]
    results: dict[str, dict[int, float]] = {a: {} for a in agents_used}

    for seed in args.seeds:
        print(f"\n#### seed={seed} ####")
        for name in agents_used:
            reward = play_episode(policies[name], name, seed, args.max_moves)
            results[name][seed] = reward

    print_table(agents_used, args.seeds, results)


if __name__ == "__main__":
    main()
