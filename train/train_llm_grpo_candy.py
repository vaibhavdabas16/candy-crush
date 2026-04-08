from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from agents.baselines import GreedyPolicy, RandomPolicy
from agents.saved_models import load_saved_policy, saved_policy_exists
from env.candy_env import CandyEnv
from utils.special_boards import SpecialInjectionConfig, reset_with_optional_specials
from utils.state_to_text import state_to_text

SWAP_RE = re.compile(r"swap\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*(?:with)?\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?", re.I)


@dataclass(frozen=True)
class Variant:
    name: str
    num_generations: int = 8
    rollout_depth: int = 1
    temperature: float = 0.85
    top_p: float = 0.95
    lr: float = 5e-6
    max_steps: int = 8
    beta_kl: float = 0.0


def parse_swap(text: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    match = SWAP_RE.search(text or "")
    if not match:
        return None
    r1, c1, r2, c2 = (int(v) for v in match.groups())
    return (r1, c1), (r2, c2)


def action_from_text(env: CandyEnv, text: str) -> int | None:
    parsed = parse_swap(text)
    if parsed is None:
        return None
    try:
        return env.encode_action(*parsed)
    except ValueError:
        return None


def prompt_for_env(env: CandyEnv) -> str:
    return (
        "Task: choose one legal Candy Crush swap. Coordinates are zero-indexed as (row,col).\n"
        "Your first line must be only the command. Do not write an intro. Do not copy the valid-action list.\n"
        "Required first-line format:\n"
        "swap (r,c) (r,c)\n"
        "Examples:\n"
        "swap (3,5) (3,6)\n"
        "swap (0,1) (1,1)\n"
        "After the first line, you may add one short reason.\n\n"
        f"{state_to_text(env, max_actions=None, include_special_rules=True)}\n"
        "Answer now. First line only the swap command:\n"
    )


def make_prompt_row(seed: int, max_moves: int, special_seed: int, special_count: int) -> dict[str, Any]:
    config = SpecialInjectionConfig(min_specials=special_count, max_specials=special_count)
    env = reset_with_optional_specials(seed, max_moves, special_seed=special_seed, special_config=config)
    return {
        "prompt": prompt_for_env(env),
        "episode_seed": seed,
        "special_seed": special_seed,
        "special_count": special_count,
        "max_moves": max_moves,
    }


def build_dataset(size: int, seed: int, max_moves: int) -> Dataset:
    rng = np.random.default_rng(seed)
    rows = []
    for idx in range(size):
        rows.append(
            make_prompt_row(
                seed=int(rng.integers(0, 10_000_000)),
                max_moves=max_moves,
                special_seed=int(rng.integers(0, 10_000_000)),
                special_count=int(rng.integers(1, 4)),
            )
        )
    return Dataset.from_list(rows)


def best_immediate_reward(env: CandyEnv) -> float:
    valid = env.valid_actions()
    if not valid:
        return float(env.invalid_penalty)
    return max(float(env.simulate_action_reward(action)) for action in valid)


def rollout_reward(env: CandyEnv, action: int, depth: int) -> float:
    if action is None or not env.is_valid_action(action):
        return float(env.invalid_penalty)
    cloned = env.clone()
    _, reward, terminated, truncated, _ = cloned.step(action)
    total = float(reward)
    for _ in range(max(0, depth - 1)):
        if terminated or truncated:
            break
        valid = cloned.valid_actions()
        if not valid:
            break
        greedy_action = max(valid, key=lambda a: cloned.simulate_action_reward(a))
        _, reward, terminated, truncated, _ = cloned.step(greedy_action)
        total += float(reward)
    return total


def candy_reward_func(
    prompts,
    completions,
    episode_seed=None,
    special_seed=None,
    special_count=None,
    max_moves=None,
    rollout_depth=None,
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    for idx, completion in enumerate(completions):
        seed = int(episode_seed[idx])
        s_seed = int(special_seed[idx])
        moves = int(max_moves[idx])
        count = int(special_count[idx])
        depth = int(rollout_depth[idx]) if rollout_depth is not None else 1
        config = SpecialInjectionConfig(min_specials=count, max_specials=count)
        env = reset_with_optional_specials(seed, moves, special_seed=s_seed, special_config=config)

        text = completion
        if isinstance(completion, list):
            text = completion[0].get("content", "") if completion else ""
        action = action_from_text(env, str(text))
        if action is None:
            rewards.append(-3.0)
            continue
        if not env.is_valid_action(action):
            rewards.append(-2.0)
            continue

        action_reward = rollout_reward(env, action, depth)
        best_reward = best_immediate_reward(env)
        denom = max(abs(best_reward), 1.0)
        quality = action_reward / denom
        best_bonus = 1.0 if math.isclose(action_reward, best_reward, rel_tol=0.0, abs_tol=1e-6) else 0.0
        first_line = str(text).strip().splitlines()[0] if str(text).strip() else ""
        format_bonus = 0.5 if SWAP_RE.fullmatch(first_line.strip()) else 0.2
        rewards.append(float(1.0 + 3.0 * quality + best_bonus + format_bonus))
    return rewards


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_lora_model(
    model_name: str,
    adapter_path: str | None,
    lora_rank: int,
    use_4bit: bool,
    beta_kl: float,
    device: str = "auto",
    dtype: str = "auto",
    is_trainable: bool = True,
):
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if dtype == "auto":
        if device == "cuda":
            torch_dtype = torch.bfloat16
        elif device == "mps":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    else:
        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]

    use_4bit = bool(use_4bit and device == "cuda")
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=is_trainable)
        if device != "cuda":
            model.to(device)
        return model

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    if device != "cuda":
        model.to(device)
    model.print_trainable_parameters()
    return model


def worker(args: argparse.Namespace) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    dataset = build_dataset(args.train_prompts, args.seed, args.max_moves)
    dataset = dataset.map(lambda row: {**row, "rollout_depth": args.rollout_depth})

    tokenizer = load_tokenizer(args.model_name)
    model = load_lora_model(
        args.model_name,
        args.init_adapter if args.init_adapter else None,
        args.lora_rank,
        not args.no_4bit,
        args.beta_kl,
    )

    train_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        bf16=True,
        tf32=True,
        beta=args.beta_kl,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    trainer = GRPOTrainer(
        model=model,
        args=train_args,
        processing_class=tokenizer,
        reward_funcs=candy_reward_func,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def generate_swap(model, tokenizer, env: CandyEnv, max_new_tokens: int, temperature: float) -> str:
    prompt = prompt_for_env(env)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def evaluate_llm_adapter(args: argparse.Namespace, adapter_path: str | None, seeds: list[int]) -> dict[str, float]:
    tokenizer = load_tokenizer(args.model_name)
    model = load_lora_model(
        args.model_name,
        adapter_path,
        args.lora_rank,
        not args.no_4bit,
        beta_kl=0.0,
        is_trainable=False,
    )
    model.eval()

    scores: list[float] = []
    invalids = 0
    cascades = 0
    for idx, seed in enumerate(seeds):
        env = reset_with_optional_specials(
            seed,
            args.max_moves,
            special_seed=seed + 50_000,
            special_config=SpecialInjectionConfig(min_specials=1, max_specials=3),
        )
        text = generate_swap(model, tokenizer, env, args.eval_max_new_tokens, args.eval_temperature)
        action = action_from_text(env, text)
        if action is None or not env.is_valid_action(action):
            reward = float(env.invalid_penalty)
            invalids += 1
        else:
            reward = float(env.simulate_action_reward(action))
        cascades += int(reward > 0)
        scores.append(reward)

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "avg_score": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "invalid_rate": float(invalids / max(1, len(seeds))),
        "cascades_per_episode": float(cascades / max(1, len(seeds))),
        "episodes": float(len(seeds)),
    }


def run_policy_episode(policy, seed: int, max_moves: int) -> tuple[float, int, int]:
    env = reset_with_optional_specials(
        seed,
        max_moves,
        special_seed=seed + 50_000,
        special_config=SpecialInjectionConfig(min_specials=1, max_specials=3),
    )
    obs, _ = env._get_obs(), env._get_info()
    module = policy.__class__.__module__
    if module.startswith("stable_baselines3") or module.startswith("sb3_contrib"):
        try:
            action, _ = policy.predict(obs, deterministic=True, action_masks=env.action_masks())
        except TypeError:
            action, _ = policy.predict(obs, deterministic=True)
    else:
        action, _ = policy.predict(obs, env=env, deterministic=True)
    action = int(action)
    if not env.is_valid_action(action):
        return float(env.invalid_penalty), 1, 0
    reward = float(env.simulate_action_reward(action))
    return reward, 0, int(reward > 0)


def evaluate_baselines(args: argparse.Namespace, seeds: list[int]) -> dict[str, dict[str, float]]:
    policies: dict[str, Any] = {"random": RandomPolicy(), "greedy": GreedyPolicy()}
    if saved_policy_exists("dqn", dqn_path=args.dqn_path):
        policies["dqn"] = load_saved_policy("dqn", dqn_path=args.dqn_path)
    if saved_policy_exists("ppo", ppo_path=args.ppo_path):
        policies["ppo"] = load_saved_policy("ppo", ppo_path=args.ppo_path, env=CandyEnv(max_moves=args.max_moves))

    results = {}
    for name, policy in policies.items():
        scores = []
        invalids = 0
        cascades = 0
        for seed in seeds:
            score, inv, cas = run_policy_episode(policy, seed, args.max_moves)
            scores.append(score)
            invalids += inv
            cascades += cas
        results[name] = {
            "avg_score": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "invalid_rate": float(invalids / max(1, len(seeds) * args.max_moves)),
            "cascades_per_episode": float(cascades / max(1, len(seeds))),
            "episodes": float(len(seeds)),
        }
    return results


def maybe_train_baselines(args: argparse.Namespace) -> None:
    if args.skip_baseline_training:
        return
    commands = []
    if not (ROOT / args.dqn_path).exists():
        commands.append([
            sys.executable,
            "train/train_dqn.py",
            "--episodes",
            str(args.dqn_episodes),
            "--model-path",
            args.dqn_path,
            "--log-path",
            "logs/dqn_rewards_llm_loop.csv",
        ])
    if not ((ROOT / args.ppo_path).exists() or Path(str(ROOT / args.ppo_path) + ".zip").exists()):
        commands.append([
            sys.executable,
            "train/train_ppo.py",
            "--timesteps",
            str(args.ppo_timesteps),
            "--model-path",
            args.ppo_path,
            "--log-path",
            "logs/ppo_rewards_llm_loop.csv",
        ])
    for command in commands:
        print("baseline_train:", " ".join(command), flush=True)
        try:
            subprocess.run(command, cwd=ROOT, timeout=args.baseline_timeout_sec, check=True)
        except subprocess.TimeoutExpired:
            print(f"baseline_timeout: {' '.join(command)}", flush=True)
        except subprocess.CalledProcessError as exc:
            print(f"baseline_failed rc={exc.returncode}: {' '.join(command)}", flush=True)


def variant_for_iteration(iteration: int, beta_kl: float) -> Variant:
    variants = [
        Variant("G8_d1_t085_lr5e6", temperature=0.85, lr=5e-6, max_steps=1, beta_kl=beta_kl),
        Variant("G8_d1_t095_lr3e6", temperature=0.95, lr=3e-6, max_steps=1, beta_kl=beta_kl),
        Variant("G8_d1_t090_lr4e6", temperature=0.90, lr=4e-6, max_steps=1, beta_kl=beta_kl),
        Variant("G8_d1_t100_lr2e6", temperature=1.00, lr=2e-6, max_steps=1, beta_kl=beta_kl),
    ]
    return variants[(iteration - 1) % len(variants)]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def write_baseline_csv(path: Path, results: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["policy", "avg_score", "score_std", "invalid_rate", "cascades_per_episode", "episodes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for policy, metrics in results.items():
            writer.writerow({"policy": policy, **metrics})


def evaluate_llm_adapter_subprocess(args: argparse.Namespace, adapter_path: str | None, seeds: list[int]) -> dict[str, float]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--eval-adapter",
        adapter_path or "",
        "--model-name",
        args.model_name,
        "--eval-seeds-csv",
        ",".join(str(seed) for seed in seeds),
        "--max-moves",
        str(args.max_moves),
        "--lora-rank",
        str(args.lora_rank),
        "--eval-max-new-tokens",
        str(args.eval_max_new_tokens),
        "--eval-temperature",
        str(args.eval_temperature),
    ]
    if args.no_4bit:
        command.append("--no-4bit")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        timeout=args.eval_timeout_sec,
        text=True,
        capture_output=True,
        check=True,
    )
    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"Could not parse eval metrics. stdout tail={completed.stdout[-1000:]}")


def supervisor(args: argparse.Namespace) -> None:
    run_dir = ROOT / args.run_dir
    candidates_dir = run_dir / "candidates"
    best_dir = run_dir / "best"
    log_path = run_dir / "experiments.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)

    maybe_train_baselines(args)
    eval_seeds = list(range(args.eval_seed, args.eval_seed + args.eval_episodes))
    baseline_results = evaluate_baselines(args, eval_seeds)
    write_baseline_csv(run_dir / "baseline_eval.csv", baseline_results)
    print("baseline_eval", json.dumps(baseline_results, sort_keys=True), flush=True)

    incumbent_adapter = str(best_dir) if best_dir.exists() else None
    if incumbent_adapter:
        incumbent = evaluate_llm_adapter_subprocess(args, incumbent_adapter, eval_seeds)
    else:
        incumbent = {"avg_score": -float("inf"), "score_std": 0.0, "invalid_rate": 1.0, "cascades_per_episode": 0.0}
    print("incumbent", json.dumps(incumbent, sort_keys=True), flush=True)

    beta_kl = args.beta_kl
    completed_iterations = 0
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    completed_iterations = max(completed_iterations, int(json.loads(line).get("iteration", 0)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

    for iteration in range(completed_iterations + 1, args.iterations + 1):
        variant = variant_for_iteration(iteration, beta_kl)
        candidate_dir = candidates_dir / f"iter_{iteration:03d}_{variant.name}"
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--model-name",
            args.model_name,
            "--output-dir",
            str(candidate_dir),
            "--seed",
            str(args.seed + iteration * 997),
            "--train-prompts",
            str(args.train_prompts),
            "--max-moves",
            str(args.max_moves),
            "--lora-rank",
            str(args.lora_rank),
            "--grad-accum",
            str(args.grad_accum),
            "--num-generations",
            str(variant.num_generations),
            "--rollout-depth",
            str(variant.rollout_depth),
            "--temperature",
            str(variant.temperature),
            "--top-p",
            str(variant.top_p),
            "--lr",
            str(variant.lr),
            "--max-steps",
            str(variant.max_steps),
            "--beta-kl",
            str(variant.beta_kl),
            "--max-completion-length",
            str(args.max_completion_length),
        ]
        if incumbent_adapter:
            command.extend(["--init-adapter", incumbent_adapter])
        if args.no_4bit:
            command.append("--no-4bit")

        started = time.time()
        status = "failed"
        error = ""
        metrics: dict[str, float] | None = None
        print(f"experiment_start iter={iteration} variant={asdict(variant)}", flush=True)
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                timeout=args.experiment_timeout_sec,
                text=True,
                capture_output=True,
            )
            status = "trained" if completed.returncode == 0 else "failed"
            error = (completed.stderr or completed.stdout)[-4000:]
            if "out of memory" in error.lower() or "cuda oom" in error.lower():
                beta_kl = 0.0
        except subprocess.TimeoutExpired as exc:
            status = "timeout"
            error = str(exc)

        if status == "trained":
            try:
                metrics = evaluate_llm_adapter_subprocess(args, str(candidate_dir), eval_seeds)
                if metrics["avg_score"] > incumbent["avg_score"]:
                    if best_dir.exists():
                        shutil.rmtree(best_dir)
                    shutil.copytree(candidate_dir, best_dir)
                    incumbent_adapter = str(best_dir)
                    incumbent = metrics
                    status = "promoted"
                else:
                    status = "discarded_non_improving"
                    shutil.rmtree(candidate_dir, ignore_errors=True)
            except RuntimeError as exc:
                status = "eval_failed"
                error = str(exc)
                shutil.rmtree(candidate_dir, ignore_errors=True)
        else:
            shutil.rmtree(candidate_dir, ignore_errors=True)
        elapsed = time.time() - started

        row = {
            "iteration": iteration,
            "status": status,
            "elapsed_sec": elapsed,
            "variant": asdict(variant),
            "metrics": metrics,
            "incumbent": incumbent,
            "baselines": baseline_results,
            "error_tail": error[-1000:],
        }
        append_jsonl(log_path, row)
        print("experiment_result", json.dumps(row, sort_keys=True), flush=True)

    if incumbent_adapter and args.final_steps > 0:
        print(f"final_training_start adapter={incumbent_adapter} steps={args.final_steps}", flush=True)
        final_dir = run_dir / "final"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--model-name",
            args.model_name,
            "--init-adapter",
            incumbent_adapter,
            "--output-dir",
            str(final_dir),
            "--seed",
            str(args.seed + 999_999),
            "--train-prompts",
            str(max(args.train_prompts, 512)),
            "--max-moves",
            str(args.max_moves),
            "--lora-rank",
            str(args.lora_rank),
            "--grad-accum",
            str(args.grad_accum),
            "--num-generations",
            "8",
            "--rollout-depth",
            "1",
            "--temperature",
            "0.9",
            "--top-p",
            "0.95",
            "--lr",
            str(args.final_lr),
            "--max-steps",
            str(args.final_steps),
            "--beta-kl",
            "0.0",
            "--max-completion-length",
            str(args.max_completion_length),
        ]
        if args.no_4bit:
            command.append("--no-4bit")
        subprocess.run(command, cwd=ROOT, check=False)
        final_metrics = evaluate_llm_adapter_subprocess(args, str(final_dir), eval_seeds)
        append_jsonl(log_path, {"status": "final", "metrics": final_metrics, "adapter": str(final_dir)})
        print("final_metrics", json.dumps(final_metrics, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--eval-adapter", type=str, default=None)
    parser.add_argument("--eval-seeds-csv", type=str, default="")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--run-dir", type=str, default="models/llm_grpo_candy")
    parser.add_argument("--output-dir", type=str, default="models/llm_grpo_candy/worker")
    parser.add_argument("--init-adapter", type=str, default="")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--experiment-timeout-sec", type=int, default=600)
    parser.add_argument("--eval-timeout-sec", type=int, default=240)
    parser.add_argument("--baseline-timeout-sec", type=int, default=600)
    parser.add_argument("--train-prompts", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-moves", type=int, default=20)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--rollout-depth", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--beta-kl", type=float, default=0.0)
    parser.add_argument("--max-completion-length", type=int, default=48)
    parser.add_argument("--eval-max-new-tokens", type=int, default=48)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--skip-baseline-training", action="store_true")
    parser.add_argument("--dqn-path", type=str, default="models/dqn.pt")
    parser.add_argument("--ppo-path", type=str, default="models/ppo")
    parser.add_argument("--dqn-episodes", type=int, default=120)
    parser.add_argument("--ppo-timesteps", type=int, default=4000)
    parser.add_argument("--final-steps", type=int, default=80)
    parser.add_argument("--final-lr", type=float, default=2e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_adapter is not None:
        seeds = [int(value) for value in args.eval_seeds_csv.split(",") if value.strip()]
        print(json.dumps(evaluate_llm_adapter(args, args.eval_adapter or None, seeds), sort_keys=True), flush=True)
    elif args.worker:
        worker(args)
    else:
        supervisor(args)


if __name__ == "__main__":
    main()
