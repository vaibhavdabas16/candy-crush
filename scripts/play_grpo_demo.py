"""Terminal demo for the Qwen GRPO GGUF agent.

Plays a full max_moves episode for each seed in --seeds and prints the
ANSI board, raw model output, and chosen action for every step. At the
end a comparison table shows GRPO vs greedy total reward per seed (and
per-policy averages), and the same data is written to
logs/grpo_comparison.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.baselines import GreedyPolicy
from agents.llm_grpo_gguf_agent import LLMGRPOGGUFAgent
from env.candy_env import CandyEnv

GGUF_REPO_ID = "arnavm7/candy-crush-qwen35-grpo-lora"
GGUF_FILENAME = "gguf/candy-crush-qwen35-grpo-Q4_K_M.gguf"
DEFAULT_GGUF_PATH = "models/llm_grpo_candy/qwen35_9b/gguf/candy-crush-qwen35-grpo-Q4_K_M.gguf"


def ensure_gguf(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    print(f"GGUF not found at {p}. Downloading from Hugging Face: {GGUF_REPO_ID}/{GGUF_FILENAME}")
    from huggingface_hub import hf_hub_download

    p.parent.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(
        repo_id=GGUF_REPO_ID,
        filename=GGUF_FILENAME,
        local_dir=str(p.parents[1]),  # .../qwen35_9b/  -> file ends up under qwen35_9b/gguf/
    )
    return Path(local)


def play_grpo_episode(agent: LLMGRPOGGUFAgent, seed: int, max_moves: int) -> float:
    env = CandyEnv(max_moves=max_moves)
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    print(f"\n----- agent=grpo_gguf  seed={seed} -----")
    for step in range(1, max_moves + 1):
        print(f"\n[step {step}]")
        print(env.render(mode="ansi"))
        action, _ = agent.predict(obs, env=env, deterministic=True)
        action = int(action)
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
    print(f"\n>>> grpo_gguf seed={seed} TOTAL REWARD = {total_reward:.2f}")
    return total_reward


def play_greedy_episode(seed: int, max_moves: int) -> float:
    env = CandyEnv(max_moves=max_moves)
    obs, _ = env.reset(seed=seed)
    pol = GreedyPolicy()
    total_reward = 0.0
    print(f"\n----- agent=greedy  seed={seed} -----")
    for step in range(1, max_moves + 1):
        print(f"\n[step {step}]")
        print(env.render(mode="ansi"))
        action, _ = pol.predict(obs, env=env, deterministic=True)
        action = int(action)
        pos_a, pos_b = env.decode_action(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        print(
            f"action={action} swap {pos_a} <-> {pos_b}  reward={reward:+.2f} "
            f"score={env.score:.2f}  moves_left={env.moves_left}",
            flush=True,
        )
        if terminated or truncated:
            break
    print(f"\n>>> greedy seed={seed} TOTAL REWARD = {total_reward:.2f}")
    return total_reward


def print_table(seeds: list[int], grpo_rewards: dict[int, float], greedy_rewards: dict[int, float]) -> None:
    width = 60
    print("\n" + "=" * width)
    print("Comparison Table  (total reward per seed)")
    print("=" * width)
    print(f"{'seed':>6s} | {'grpo_gguf':>12s} | {'greedy':>10s}")
    print("-" * width)
    for s in seeds:
        print(f"{s:>6d} | {grpo_rewards[s]:>12.2f} | {greedy_rewards[s]:>10.2f}")
    print("-" * width)
    g_avg = sum(grpo_rewards.values()) / len(grpo_rewards)
    gd_avg = sum(greedy_rewards.values()) / len(greedy_rewards)
    print(f"{'avg':>6s} | {g_avg:>12.2f} | {gd_avg:>10.2f}")
    print("=" * width)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max-moves", type=int, default=20)
    p.add_argument("--gguf-path", default=DEFAULT_GGUF_PATH)
    p.add_argument("--gguf-n-ctx", type=int, default=4096)
    p.add_argument("--gguf-n-threads", type=int, default=None)
    p.add_argument(
        "--gguf-n-gpu-layers",
        type=int,
        default=0,
        help="-1 = offload all layers to GPU (Metal/CUDA); 0 = CPU only (default).",
    )
    p.add_argument("--gguf-max-new-tokens", type=int, default=24)
    p.add_argument("--no-greedy", action="store_true", help="Skip the greedy comparison column.")
    p.add_argument("--log-dir", default="logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    gguf_path = ensure_gguf(args.gguf_path)

    print(
        f"\nGRPO demo: seeds={args.seeds}  max_moves={args.max_moves}  gguf={gguf_path}"
        f"  n_gpu_layers={args.gguf_n_gpu_layers}\n"
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
        log_io=True,
        no_fallback=True,
    )

    grpo_rewards: dict[int, float] = {}
    greedy_rewards: dict[int, float] = {}

    for seed in args.seeds:
        print(f"\n#### seed={seed} ####")
        grpo_rewards[seed] = play_grpo_episode(agent, seed, args.max_moves)
        if not args.no_greedy:
            greedy_rewards[seed] = play_greedy_episode(seed, args.max_moves)

    if args.no_greedy:
        # Fall back to a single-column table.
        width = 40
        print("\n" + "=" * width)
        print("Comparison Table  (grpo_gguf only)")
        print("=" * width)
        print(f"{'seed':>6s} | {'grpo_gguf':>12s}")
        print("-" * width)
        for s in args.seeds:
            print(f"{s:>6d} | {grpo_rewards[s]:>12.2f}")
        avg = sum(grpo_rewards.values()) / len(grpo_rewards)
        print("-" * width)
        print(f"{'avg':>6s} | {avg:>12.2f}")
        print("=" * width)
    else:
        print_table(args.seeds, grpo_rewards, greedy_rewards)

    print(
        f"\ngrpo_gguf parse_failures={agent.parse_failures}  "
        f"invalid_actions={agent.invalid_actions}"
    )


if __name__ == "__main__":
    main()
