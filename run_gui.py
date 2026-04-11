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
        choices=["manual", "random", "greedy", "dqn", "ppo", "grpo", "llm_grpo", "llm_grpo_gguf"],
        default="manual",
    )
    parser.add_argument("--max-moves", type=int, default=20)
    parser.add_argument("--dqn-path",  type=str, default="models/dqn.pt")
    parser.add_argument("--ppo-path",  type=str, default="models/ppo")
    parser.add_argument("--grpo-path", type=str, default="models/grpo_candy.pt")
    parser.add_argument("--llm-grpo-path", type=str, default="models/llm_grpo_candy/qwen35_9b/final_plus30")
    parser.add_argument("--llm-model-name", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--llm-no-4bit", action="store_true")
    parser.add_argument("--llm-device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--llm-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--llm-max-new-tokens", type=int, default=32)
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
        help="-1 = offload all layers to Metal/GPU; 0 = CPU only.",
    )
    parser.add_argument(
        "--gguf-log-io",
        action="store_true",
        help="Print model raw output, parsed action, and chosen action per step.",
    )
    parser.add_argument(
        "--gguf-log-prompt",
        action="store_true",
        help="Also print the full prompt fed to the model each step (verbose).",
    )
    parser.add_argument("--agent-delay", type=float, default=0.35)
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run the episode in the terminal (no Pygame). Prints board state per step.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Reset seed for terminal mode.")
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
        not args.llm_no_4bit,
        args.llm_device,
        args.llm_dtype,
        args.llm_max_new_tokens,
        gguf_path=ROOT / args.gguf_path if not Path(args.gguf_path).is_absolute() else args.gguf_path,
        gguf_n_ctx=args.gguf_n_ctx,
        gguf_n_threads=args.gguf_n_threads,
        gguf_n_gpu_layers=args.gguf_n_gpu_layers,
        gguf_log_io=args.gguf_log_io,
        gguf_log_prompt=args.gguf_log_prompt,
    )
    if args.no_gui:
        run_terminal(env, policy, args)
        return

    viewer = CandyViewer(env, policy=policy, mode=args.agent)
    viewer.config.agent_delay = args.agent_delay
    viewer.run()


def run_terminal(env: CandyEnv, policy, args: argparse.Namespace) -> None:
    if args.agent == "manual":
        raise SystemExit("--no-gui requires an agent (e.g. --agent llm_grpo_gguf)")

    obs, _ = env.reset(seed=args.seed)
    total_reward = 0.0
    for step in range(1, args.max_moves + 1):
        print(f"\n=== step {step} ===")
        print(env.render(mode="ansi"))

        try:
            action, _ = policy.predict(obs, env=env, deterministic=True)
        except TypeError:
            try:
                action, _ = policy.predict(obs, deterministic=True, action_masks=env.action_masks())
            except TypeError:
                action, _ = policy.predict(obs, deterministic=True)
        action = int(action)

        pos_a, pos_b = env.decode_action(action)
        obs, reward, terminated, truncated, _info = env.step(action)
        total_reward += float(reward)
        print(
            f"action={action} swap {pos_a} <-> {pos_b}  reward={reward:.2f}  "
            f"score={env.score:.2f}  moves_left={env.moves_left}",
            flush=True,
        )
        if terminated or truncated:
            print("\n[episode ended]")
            break

    print(f"\nFinal score: {env.score:.2f}   total reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()
