"""Fixed-board evaluation for Candy Crush policies.

Plays a full max_moves rollout on each of N fixed special-candy boards and
reports per-board reward, mean/std/min/max, invalid action rate, and (for
the LLM GRPO GGUF policy) parse-invalid rate.

Default protocol matches the GRPO eval spec: 10 boards with seeds
20000..20009, special_seed = seed + 50000, max_moves = 20.

Examples:
    python eval/evaluate.py
    python eval/evaluate.py --policies random greedy dqn ppo grpo_gguf
    python eval/evaluate.py --policies grpo_gguf --gguf-n-gpu-layers -1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from agents.baselines import GreedyPolicy, RandomPolicy
from agents.saved_models import load_saved_policy, saved_policy_exists
from env.candy_env import CandyEnv
from utils.special_boards import SpecialInjectionConfig, inject_random_specials

DEFAULT_SEEDS = list(range(20000, 20010))
DEFAULT_SPECIAL_OFFSET = 50000
ALL_POLICIES = ["random", "greedy", "dqn", "ppo", "grpo_gguf"]


def make_env(seed: int, special_seed: int, max_moves: int) -> tuple[CandyEnv, np.ndarray]:
    """Build a CandyEnv reset to the same fixed special-board, return (env, obs)."""
    env = CandyEnv(max_moves=max_moves)
    obs, _ = env.reset(seed=seed)
    if special_seed is not None:
        inject_random_specials(env, special_seed, SpecialInjectionConfig())
    return env, obs


def policy_predict(policy, obs, env: CandyEnv) -> int:
    module = policy.__class__.__module__
    if module.startswith("stable_baselines3") or module.startswith("sb3_contrib"):
        try:
            action, _ = policy.predict(obs, deterministic=True, action_masks=env.action_masks())
        except TypeError:
            action, _ = policy.predict(obs, deterministic=True)
        return int(action)
    try:
        action, _ = policy.predict(obs, env=env, deterministic=True)
    except TypeError:
        action, _ = policy.predict(obs, deterministic=True)
    return int(action)


def run_rollout(policy, seed: int, special_seed: int, max_moves: int) -> dict:
    """Play a full max_moves rollout on one fixed board."""
    env, obs = make_env(seed=seed, special_seed=special_seed, max_moves=max_moves)
    total_reward = 0.0
    invalid_steps = 0
    for _ in range(max_moves):
        action = policy_predict(policy, obs, env)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        if info.get("invalid", False):
            invalid_steps += 1
        if terminated or truncated:
            break
    return {
        "reward": total_reward,
        "invalid_steps": invalid_steps,
        "score": float(env.score),
    }


def eval_policy(
    name: str,
    policy,
    seeds: list[int],
    special_offset: int,
    max_moves: int,
    *,
    grpo_agent=None,
) -> dict:
    per_board = []
    invalid_total = 0
    move_total = 0
    parse_failures_start = grpo_agent.parse_failures if grpo_agent is not None else 0
    invalid_actions_start = grpo_agent.invalid_actions if grpo_agent is not None else 0

    t0 = time.time()
    for s in seeds:
        result = run_rollout(policy, seed=s, special_seed=s + special_offset, max_moves=max_moves)
        per_board.append({"seed": s, "reward": result["reward"], "score": result["score"]})
        invalid_total += result["invalid_steps"]
        move_total += max_moves
        print(
            f"  [{name:>10s}] seed={s} reward={result['reward']:8.2f} "
            f"score={result['score']:8.2f} invalid_steps={result['invalid_steps']}",
            flush=True,
        )
    elapsed = time.time() - t0

    rewards = [b["reward"] for b in per_board]
    summary = {
        "policy": name,
        "per_board": per_board,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards, ddof=0)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "invalid_rate": invalid_total / move_total if move_total else 0.0,
        "boards": len(seeds),
        "elapsed_s": elapsed,
    }

    if grpo_agent is not None:
        parse_fails = grpo_agent.parse_failures - parse_failures_start
        invalid_acts = grpo_agent.invalid_actions - invalid_actions_start
        summary["grpo_parse_invalid_rate"] = parse_fails / move_total if move_total else 0.0
        summary["grpo_model_invalid_rate"] = invalid_acts / move_total if move_total else 0.0
        summary["grpo_parse_failures"] = parse_fails
        summary["grpo_invalid_actions"] = invalid_acts

    return summary


def print_summary(summaries: list[dict]) -> None:
    print()
    print("=" * 92)
    print("Summary  (avg ± std  [min, max]   invalid-rate   boards)")
    print("=" * 92)
    for s in summaries:
        line = (
            f"{s['policy']:>10s} | avg={s['avg_reward']:8.2f} ± {s['std_reward']:7.2f} "
            f"| min={s['min_reward']:8.2f} | max={s['max_reward']:8.2f} "
            f"| invalid={s['invalid_rate']:.3f} | n={s['boards']} "
            f"| {s['elapsed_s']:.1f}s"
        )
        if "grpo_parse_invalid_rate" in s:
            line += (
                f"\n           | parse_invalid={s['grpo_parse_invalid_rate']:.3f} "
                f"| model_invalid={s['grpo_model_invalid_rate']:.3f}"
            )
        print(line)
    print("=" * 92)
    print()
    print("Per-board rewards")
    header = f"{'seed':>6s} | " + " | ".join(f"{s['policy']:>10s}" for s in summaries)
    print(header)
    print("-" * len(header))
    seeds = [b["seed"] for b in summaries[0]["per_board"]]
    for i, seed in enumerate(seeds):
        row = f"{seed:>6d} | " + " | ".join(
            f"{s['per_board'][i]['reward']:>10.2f}" for s in summaries
        )
        print(row)


def build_policy(name: str, args: argparse.Namespace, env: CandyEnv):
    if name == "random":
        return RandomPolicy(), None
    if name == "greedy":
        return GreedyPolicy(), None
    if name == "dqn":
        if not saved_policy_exists("dqn", dqn_path=args.dqn_path):
            raise FileNotFoundError(f"DQN model not found: {ROOT / args.dqn_path}")
        return load_saved_policy("dqn", dqn_path=args.dqn_path), None
    if name == "ppo":
        if not saved_policy_exists("ppo", ppo_path=args.ppo_path):
            raise FileNotFoundError(f"PPO model not found: {ROOT / args.ppo_path}.zip")
        return load_saved_policy("ppo", ppo_path=args.ppo_path, env=env), None
    if name == "grpo_gguf":
        from agents.llm_grpo_gguf_agent import LLMGRPOGGUFAgent

        gguf_path = Path(args.gguf_path)
        if not gguf_path.is_absolute():
            gguf_path = ROOT / gguf_path
        if not gguf_path.exists():
            raise FileNotFoundError(
                f"GGUF model not found: {gguf_path}\n"
                "Download with:\n"
                "  huggingface-cli download arnavm7/candy-crush-qwen35-grpo-lora \\\n"
                "    gguf/candy-crush-qwen35-grpo-Q4_K_M.gguf \\\n"
                "    gguf/candy-crush-qwen35-grpo-Q4_K_M.gguf.sha256 \\\n"
                "    --local-dir models/llm_grpo_candy/qwen35_9b"
            )
        agent = LLMGRPOGGUFAgent(
            gguf_path,
            n_ctx=args.gguf_n_ctx,
            n_threads=args.gguf_n_threads,
            n_gpu_layers=args.gguf_n_gpu_layers,
            max_new_tokens=args.gguf_max_new_tokens,
            temperature=0.0,
            seed=0,
            verbose=False,
            log_io=args.gguf_log_io,
            no_fallback=True,  # eval spec: count invalids, no greedy rescue
        )
        return agent, agent
    raise ValueError(f"Unknown policy: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policies", nargs="+", default=ALL_POLICIES, choices=ALL_POLICIES)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--special-offset", type=int, default=DEFAULT_SPECIAL_OFFSET)
    parser.add_argument("--max-moves", type=int, default=20)
    parser.add_argument("--dqn-path", type=str, default="models/dqn.pt")
    parser.add_argument("--ppo-path", type=str, default="models/ppo")
    parser.add_argument(
        "--gguf-path",
        type=str,
        default="models/llm_grpo_candy/qwen35_9b/gguf/candy-crush-qwen35-grpo-Q4_K_M.gguf",
    )
    parser.add_argument("--gguf-n-ctx", type=int, default=4096)
    parser.add_argument("--gguf-n-threads", type=int, default=None)
    parser.add_argument(
        "--gguf-n-gpu-layers",
        type=int,
        default=-1,
        help="-1 = offload all layers to GPU (Metal/CUDA); 0 = CPU only.",
    )
    parser.add_argument("--gguf-max-new-tokens", type=int, default=24)
    parser.add_argument("--gguf-log-io", action="store_true")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to dump full results as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"Eval protocol: boards={len(args.seeds)} max_moves={args.max_moves} "
        f"special_offset={args.special_offset}\n"
        f"Seeds: {args.seeds}\n"
        f"Policies: {args.policies}\n"
    )

    env_for_init = CandyEnv(max_moves=args.max_moves)

    summaries = []
    for name in args.policies:
        print(f"\n>>> Evaluating: {name}")
        try:
            policy, grpo_agent = build_policy(name, args, env_for_init)
        except FileNotFoundError as e:
            print(f"  skipped: {e}")
            continue
        summary = eval_policy(
            name,
            policy,
            seeds=args.seeds,
            special_offset=args.special_offset,
            max_moves=args.max_moves,
            grpo_agent=grpo_agent,
        )
        summaries.append(summary)
        if hasattr(policy, "close"):
            try:
                policy.close()
            except Exception:
                pass

    if not summaries:
        print("No policies were evaluated.")
        return

    print_summary(summaries)

    if args.json_out:
        import json

        out_path = Path(args.json_out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(
                {
                    "config": {
                        "seeds": args.seeds,
                        "special_offset": args.special_offset,
                        "max_moves": args.max_moves,
                        "policies": args.policies,
                    },
                    "results": summaries,
                },
                f,
                indent=2,
            )
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
