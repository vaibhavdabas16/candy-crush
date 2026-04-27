from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from env.candy_env import CandyEnv
from utils.state_to_text import state_to_text

SWAP_RE = re.compile(
    r"swap\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*(?:with)?\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?",
    re.I,
)


def _parse_swap(text: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    m = SWAP_RE.search(text or "")
    if not m:
        return None
    r1, c1, r2, c2 = (int(v) for v in m.groups())
    return (r1, c1), (r2, c2)


def _action_from_text(env: CandyEnv, text: str) -> int | None:
    parsed = _parse_swap(text)
    if parsed is None:
        return None
    try:
        return env.encode_action(*parsed)
    except ValueError:
        return None


def _pick_invalid_action(env: CandyEnv) -> int:
    """Pick any action that is *not* legal on the current board.

    Used when GRPO is run with no_fallback=True and the model output is
    unparsable. Guarantees the env will apply invalid_penalty.
    """
    valid = set(int(a) for a in env.valid_actions())
    n = int(env.action_space.n)
    for a in range(n):
        if a not in valid:
            return a
    return 0  # extreme edge case: every action valid


def _prompt_for_env(env: CandyEnv) -> str:
    return (
        "Task: choose one legal Candy Crush swap. Coordinates are zero-indexed as (row,col).\n"
        "Your first line must be only the command. Do not write an intro. Do not copy the valid-action list.\n"
        "Required first-line format:\n"
        "swap (r,c) (r,c)\n"
        "Examples:\n"
        "swap (3,5) (3,6)\n"
        "swap (0,1) (1,1)\n"
        "After the first line, you should add one short reason.\n\n"
        f"{state_to_text(env, max_actions=None, include_special_rules=True)}\n"
        "Answer now. First line only the swap command:\n"
    )


class LLMGRPOGGUFAgent:
    """llama.cpp-backed Qwen GRPO Candy Crush policy.

    Loads the merged Q4_K_M GGUF directly via llama-cpp-python, so it does
    not need the base model + LoRA adapter. Suitable for Mac/CPU and
    integrates with the Pygame GUI via the same predict() contract as
    LLMGRPOAgent.
    """

    def __init__(
        self,
        gguf_path: str | Path,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        n_gpu_layers: int = -1,
        max_new_tokens: int = 80,
        temperature: float = 0.0,
        seed: int = 0,
        verbose: bool = False,
        log_io: bool = False,
        log_prompt: bool = False,
        no_fallback: bool = False,
    ) -> None:
        from llama_cpp import Llama

        self.gguf_path = Path(gguf_path)
        if not self.gguf_path.exists():
            raise FileNotFoundError(f"GGUF model not found: {self.gguf_path}")

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.log_io = log_io
        self.log_prompt = log_prompt
        self.no_fallback = no_fallback
        self._step = 0
        self.parse_failures = 0
        self.invalid_actions = 0
        self.last_raw = ""
        self.last_parsed_action: int | None = None
        self.last_was_valid = False

        self.llm = Llama(
            model_path=str(self.gguf_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=verbose,
        )

    def generate(self, prompt: str, deterministic: bool = True) -> str:
        out = self.llm.create_completion(
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            temperature=0.0 if deterministic else self.temperature,
            stop=["\n\n"],
        )
        return out["choices"][0]["text"]

    def predict(
        self,
        obs: np.ndarray,
        env: CandyEnv | None = None,
        deterministic: bool = True,
    ) -> tuple[int, None]:
        if env is None:
            raise ValueError("LLMGRPOGGUFAgent requires env so it can serialize the board state.")

        import time

        self._step += 1
        prompt = _prompt_for_env(env)

        if self.log_io and self.log_prompt:
            print(
                f"\n========== grpo-gguf step={self._step} prompt ==========\n"
                f"{prompt}"
                f"========================================================",
                flush=True,
            )

        t0 = time.time()
        text = self.generate(prompt, deterministic=deterministic)
        gen_s = time.time() - t0

        action = _action_from_text(env, text)
        parse_failed = action is None
        valid_action = (action is not None) and env.is_valid_action(int(action))

        self.last_raw = text
        self.last_parsed_action = action
        self.last_was_valid = bool(valid_action)
        if parse_failed:
            self.parse_failures += 1
        if not valid_action:
            self.invalid_actions += 1

        if self.no_fallback:
            if action is not None:
                chosen = int(action)
                tag = "ok" if valid_action else "model-invalid"
            else:
                chosen = _pick_invalid_action(env)
                tag = "parse-fail"
            fallback = None
        else:
            chosen = int(action) if valid_action else None
            fallback = None
            if not valid_action:
                valid = env.valid_actions()
                if valid:
                    fallback = max(valid, key=lambda a: env.simulate_action_reward(int(a)))
                else:
                    fallback = int(env.action_space.sample())
                chosen = fallback
            tag = "ok" if valid_action else ("fallback-greedy" if fallback is not None else "random")

        if self.log_io:
            decoded = env.decode_action(int(chosen)) if chosen is not None else None
            print(
                f"[grpo-gguf step={self._step} {gen_s:.2f}s] "
                f"raw={text.strip()!r} "
                f"parsed_action={action} "
                f"valid={valid_action} "
                f"chosen={chosen} {decoded} [{tag}]",
                flush=True,
            )

        return int(chosen), None

    def close(self) -> None:
        del self.llm
