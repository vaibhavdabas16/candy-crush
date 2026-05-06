"""Head-to-head benchmark: GRPO GGUF vs an OpenAI Chat Completions model.

Both policies are run on the same fixed special-candy boards (the GRPO
eval protocol: seeds 20000-20009, special_seed = seed + 50000, 20-move
rollouts). Both use the same prompt, the same parser, and the same
no_fallback contract, so the comparison is fair.

The OpenAI key is read from OPENAI_API_KEY at run time. It is never
accepted on the command line and never written to disk.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from agents.llm_grpo_gguf_agent import LLMGRPOGGUFAgent
from agents.openai_agent import OpenAIAgent
from env.candy_env import CandyEnv
from utils.special_boards import SpecialInjectionConfig, inject_random_specials


DEFAULT_SEEDS = list(range(20000, 20010))
DEFAULT_SPECIAL_OFFSET = 50000
DEFAULT_GGUF_PATH = "models/llm_grpo_candy/qwen35_9b/gguf/candy-crush-qwen35-grpo-Q4_K_M.gguf"


def make_env(seed: int, special_seed: int, max_moves: int) -> tuple[CandyEnv, np.ndarray]:
    env = CandyEnv(max_moves=max_moves)
    obs, _ = env.reset(seed=seed)
    inject_random_specials(env, special_seed, SpecialInjectionConfig())
    return env, obs


def play(agent, label: str, seed: int, special_seed: int, max_moves: int) -> dict:
    env, obs = make_env(seed=seed, special_seed=special_seed, max_moves=max_moves)
    total_reward = 0.0
    invalid_steps = 0
    t0 = time.time()
    for _ in range(max_moves):
        action, _ = agent.predict(obs, env=env, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        if info.get("invalid", False):
            invalid_steps += 1
        if terminated or truncated:
            break
    elapsed = time.time() - t0
    print(
        f"  [{label:>14s}] seed={seed} reward={total_reward:8.2f} "
        f"score={env.score:8.2f} invalid_steps={invalid_steps} "
        f"({elapsed:.1f}s)",
        flush=True,
    )
    return {"reward": total_reward, "score": float(env.score), "invalid_steps": invalid_steps}


def summarize(label: str, per_seed: list[dict], parse_failures: int, invalid_actions: int, n_decisions: int) -> dict:
    rewards = [r["reward"] for r in per_seed]
    return {
        "policy": label,
        "avg": float(np.mean(rewards)),
        "std": float(np.std(rewards, ddof=0)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "invalid_steps": sum(r["invalid_steps"] for r in per_seed),
        "parse_failures": parse_failures,
        "invalid_actions": invalid_actions,
        "decisions": n_decisions,
        "rewards": rewards,
    }


def print_table(seeds: list[int], rows: list[dict]) -> None:
    width = 80
    print("\n" + "=" * width)
    print(f"Head-to-head: {' vs '.join(r['policy'] for r in rows)}")
    print("=" * width)
    print(f"{'Policy':>16s} | {'avg ± std':>20s} | {'min':>6s} | {'max':>6s} | {'parse-fail':>10s} | {'model-inv':>9s}")
    print("-" * width)
    for r in rows:
        pf_rate = r["parse_failures"] / r["decisions"] if r["decisions"] else 0.0
        inv_rate = r["invalid_actions"] / r["decisions"] if r["decisions"] else 0.0
        print(
            f"{r['policy']:>16s} | "
            f"{r['avg']:8.2f} ± {r['std']:7.2f} | "
            f"{r['min']:6.0f} | {r['max']:6.0f} | "
            f"{pf_rate:>9.3f}  | {inv_rate:>8.3f}"
        )
    print()
    print("Per-board reward")
    print(f"{'seed':>6s} | " + " | ".join(f"{r['policy']:>16s}" for r in rows))
    print("-" * (10 + 19 * len(rows)))
    for i, s in enumerate(seeds):
        cells = " | ".join(f"{r['rewards'][i]:>16.2f}" for r in rows)
        print(f"{s:>6d} | {cells}")
    print("=" * width)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--special-offset", type=int, default=DEFAULT_SPECIAL_OFFSET)
    p.add_argument("--max-moves", type=int, default=20)
    p.add_argument("--gguf-path", type=str, default=DEFAULT_GGUF_PATH)
    p.add_argument("--gguf-n-ctx", type=int, default=4096)
    p.add_argument("--gguf-n-gpu-layers", type=int, default=-1, help="-1 = all layers offloaded; 0 = CPU only.")
    p.add_argument("--gguf-max-new-tokens", type=int, default=24)
    p.add_argument("--gguf-log-io", action="store_true")
    p.add_argument("--openai-model", type=str, default="gpt-5")
    p.add_argument("--openai-max-tokens", type=int, default=4096)
    p.add_argument("--openai-log-io", action="store_true")
    p.add_argument("--skip-grpo", action="store_true")
    p.add_argument("--skip-openai", action="store_true")
    p.add_argument("--json-out", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_openai and not os.environ.get("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY is not set in the environment.\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "Or pass --skip-openai to run only the GRPO policy.",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"Seeds: {args.seeds}  max_moves={args.max_moves}")
    rows = []

    if not args.skip_grpo:
        print(f"\n>>> GRPO GGUF policy ({args.gguf_path})")
        gguf_path = Path(args.gguf_path)
        if not gguf_path.is_absolute():
            gguf_path = ROOT / gguf_path
        grpo = LLMGRPOGGUFAgent(
            gguf_path,
            n_ctx=args.gguf_n_ctx,
            n_gpu_layers=args.gguf_n_gpu_layers,
            max_new_tokens=args.gguf_max_new_tokens,
            temperature=0.0,
            log_io=args.gguf_log_io,
            no_fallback=True,
        )
        per_seed = [
            play(grpo, "grpo_gguf", s, s + args.special_offset, args.max_moves)
            for s in args.seeds
        ]
        rows.append(
            summarize(
                "grpo_gguf",
                per_seed,
                grpo.parse_failures,
                grpo.invalid_actions,
                grpo._step,
            )
        )

    if not args.skip_openai:
        print(f"\n>>> OpenAI policy ({args.openai_model})")
        oai = OpenAIAgent(
            model=args.openai_model,
            max_tokens=args.openai_max_tokens,
            temperature=0.0,
            log_io=args.openai_log_io,
            no_fallback=True,
        )
        per_seed = [
            play(oai, f"openai/{args.openai_model}", s, s + args.special_offset, args.max_moves)
            for s in args.seeds
        ]
        rows.append(
            summarize(
                f"openai/{args.openai_model}",
                per_seed,
                oai.parse_failures,
                oai.invalid_actions,
                oai._step,
            )
        )

    if not rows:
        print("Nothing to evaluate (both --skip-grpo and --skip-openai).")
        return

    print_table(args.seeds, rows)

    if args.json_out:
        import json

        out = Path(args.json_out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(
                {
                    "config": {
                        "seeds": args.seeds,
                        "special_offset": args.special_offset,
                        "max_moves": args.max_moves,
                        "gguf_path": str(args.gguf_path),
                        "openai_model": args.openai_model,
                    },
                    "results": rows,
                },
                f,
                indent=2,
            )
        print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
