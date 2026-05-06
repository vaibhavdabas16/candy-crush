"""Gemini policy for Candy Crush.

Same prompt + parser + no_fallback contract as the GRPO and OpenAI
agents so the comparison is apples-to-apples. The API key is read
from GEMINI_API_KEY (or GOOGLE_API_KEY) at runtime - never accepted
on the command line, never written to disk.
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


class GeminiAgent:
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        log_io: bool = False,
        log_prompt: bool = False,
        no_fallback: bool = True,
        timeout: float = 60.0,
    ) -> None:
        try:
            from google import genai  # google-genai SDK
        except ImportError as e:
            raise ImportError(
                "google-genai package not installed. "
                "Run: pip install 'google-genai>=0.3'"
            ) from e

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. "
                "Export it before running, e.g. export GEMINI_API_KEY=..."
            )

        self.client = genai.Client(api_key=key)
        self.model = model
        self.max_tokens = max_tokens
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

    def _generate(self, prompt: str) -> str:
        from google.genai import types

        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        return resp.text or ""

    def predict(
        self,
        obs: np.ndarray,
        env: CandyEnv | None = None,
        deterministic: bool = True,
    ) -> tuple[int, None]:
        if env is None:
            raise ValueError("GeminiAgent requires env so it can serialize the board state.")

        self._step += 1
        prompt = _prompt_for_env(env)

        if self.log_io and self.log_prompt:
            print(
                f"\n========== gemini step={self._step} prompt ==========\n"
                f"{prompt}"
                f"=====================================================",
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
                f"[gemini({self.model}) step={self._step} {gen_s:.2f}s] "
                f"raw={text.strip()!r} "
                f"parsed_action={action} "
                f"valid={valid_action} "
                f"chosen={chosen} {decoded} [{tag}]",
                flush=True,
            )

        return int(chosen), None
