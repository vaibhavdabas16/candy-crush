"""Stand-alone Akash ML benchmark on the GRPO eval boards."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from agents.akash_agent import AkashAgent
from env.candy_env import CandyEnv
from utils.special_boards import SpecialInjectionConfig, inject_random_specials

DEFAULT_SEEDS = list(range(20000, 20010))
DEFAULT_SPECIAL_OFFSET = 50000


def make_env(seed: int, special_seed: int, max_moves: int):
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
    print(
        f"  [{label:>14s}] seed={seed} reward={total_reward:8.2f} "
        f"score={env.score:8.2f} invalid_steps={invalid_steps} ({time.time()-t0:.1f}s)",
        flush=True,
    )
    return {"reward": total_reward, "invalid_steps": invalid_steps}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--base-url", default="https://api.akashml.com/v1")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--special-offset", type=int, default=DEFAULT_SPECIAL_OFFSET)
    p.add_argument("--max-moves", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--log-io", action="store_true")
    p.add_argument("--json-out", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (os.environ.get("AKASH_ML_API_KEY") or os.environ.get("AKASHML_API_KEY")):
        print("ERROR: AKASH_ML_API_KEY not set in env.", file=sys.stderr)
        sys.exit(2)

    print(f"Akash benchmark: model={args.model} seeds={args.seeds} max_moves={args.max_moves}")
    agent = AkashAgent(
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        log_io=args.log_io,
        no_fallback=True,
    )
    per_seed = [play(agent, f"akash/{args.model}", s, s + args.special_offset, args.max_moves) for s in args.seeds]

    rewards = [r["reward"] for r in per_seed]
    avg = float(np.mean(rewards))
    std = float(np.std(rewards, ddof=0))
    print()
    print("=" * 80)
    print(f"Summary  policy=akash/{args.model}")
    print("=" * 80)
    print(f"avg ± std  : {avg:.2f} ± {std:.2f}")
    print(f"min / max  : {min(rewards):.0f} / {max(rewards):.0f}")
    print(f"parse-fail : {agent.parse_failures}/{agent._step}")
    print(f"model-inv  : {agent.invalid_actions}/{agent._step}")

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
                        "model": args.model,
                        "base_url": args.base_url,
                        "seeds": args.seeds,
                        "special_offset": args.special_offset,
                        "max_moves": args.max_moves,
                    },
                    "rewards": rewards,
                    "avg": avg,
                    "std": std,
                    "min": min(rewards),
                    "max": max(rewards),
                    "parse_failures": agent.parse_failures,
                    "invalid_actions": agent.invalid_actions,
                    "decisions": agent._step,
                },
                f,
                indent=2,
            )
        print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
