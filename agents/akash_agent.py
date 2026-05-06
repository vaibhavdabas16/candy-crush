"""Akash ML chat-completions policy.

Akash exposes an OpenAI-compatible /v1/chat/completions endpoint at
https://api.akashml.com so we can reuse the OpenAI SDK with a custom
base_url. The API key is read from AKASH_ML_API_KEY at runtime.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from agents.llm_grpo_gguf_agent import (
    _action_from_text,
    _pick_invalid_action,
    _prompt_for_env,
)
from env.candy_env import CandyEnv

DEFAULT_BASE_URL = "https://api.akashml.com/v1"


class AkashAgent:
    def __init__(
        self,
        model: str = "Qwen/Qwen3.6-35B-A3B",
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.9,
        log_io: bool = False,
        log_prompt: bool = False,
        no_fallback: bool = True,
        timeout: float = 120.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Run: pip install 'openai>=1.40'") from e

        key = api_key or os.environ.get("AKASH_ML_API_KEY") or os.environ.get("AKASHML_API_KEY")
        if not key:
            raise RuntimeError(
                "AKASH_ML_API_KEY not set. Export it before running, "
                "e.g. export AKASH_ML_API_KEY=..."
            )

        self.client = OpenAI(api_key=key, base_url=base_url, timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.log_io = log_io
        self.log_prompt = log_prompt
        self.no_fallback = no_fallback

        self._step = 0
        self.parse_failures = 0
        self.invalid_actions = 0
        self.last_raw = ""
        self.last_parsed_action: int | None = None
        self.last_was_valid = False

    def _generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        return resp.choices[0].message.content or ""

    def predict(
        self,
        obs: np.ndarray,
        env: CandyEnv | None = None,
        deterministic: bool = True,
    ) -> tuple[int, None]:
        if env is None:
            raise ValueError("AkashAgent requires env so it can serialize the board state.")

        self._step += 1
        prompt = _prompt_for_env(env)

        if self.log_io and self.log_prompt:
            print(
                f"\n========== akash step={self._step} prompt ==========\n"
                f"{prompt}"
                f"====================================================",
                flush=True,
            )

        t0 = time.time()
        text = self._generate(prompt)
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
        else:
            chosen = int(action) if valid_action else None
            if not valid_action:
                valid = env.valid_actions()
                if valid:
                    chosen = max(valid, key=lambda a: env.simulate_action_reward(int(a)))
                else:
                    chosen = int(env.action_space.sample())
                tag = "fallback-greedy"
            else:
                tag = "ok"

        if self.log_io:
            decoded = env.decode_action(int(chosen))
            print(
                f"[akash({self.model}) step={self._step} {gen_s:.2f}s] "
                f"raw={text.strip()!r} "
                f"parsed_action={action} "
                f"valid={valid_action} "
                f"chosen={chosen} {decoded} [{tag}]",
                flush=True,
            )

        return int(chosen), None
