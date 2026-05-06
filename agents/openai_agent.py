"""OpenAI Chat-Completions policy for Candy Crush.

Uses the same prompt and the same no_fallback contract as the GRPO GGUF
agent so that head-to-head numbers are an apples-to-apples comparison.
The API key is read from the OPENAI_API_KEY environment variable - it
is never accepted on the command line and never written to disk.
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


class OpenAIAgent:
    def __init__(
        self,
        model: str = "gpt-5",
        api_key: str | None = None,
        max_tokens: int = 4096,  # GPT-5+ reasoning models need ample budget
        temperature: float = 0.0,
        log_io: bool = False,
        log_prompt: bool = False,
        no_fallback: bool = True,
        timeout: float = 60.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package not installed. Run: pip install 'openai>=1.40'"
            ) from e

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it in the shell before "
                "running, e.g.  export OPENAI_API_KEY=sk-..."
            )

        self.client = OpenAI(api_key=key, timeout=timeout)
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
        # GPT-5+ reasoning models reject `max_tokens` and `temperature`.
        # Try the new `max_completion_tokens` first; if that errors out
        # we fall back to the legacy parameter for older models.
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            resp = self.client.chat.completions.create(
                **kwargs, max_completion_tokens=self.max_tokens
            )
        except Exception as e:
            msg = str(e).lower()
            if "max_completion_tokens" in msg or "unsupported_parameter" in msg:
                resp = self.client.chat.completions.create(
                    **kwargs, max_tokens=self.max_tokens, temperature=self.temperature
                )
            else:
                raise
        return resp.choices[0].message.content or ""

    def predict(
        self,
        obs: np.ndarray,
        env: CandyEnv | None = None,
        deterministic: bool = True,
    ) -> tuple[int, None]:
        if env is None:
            raise ValueError("OpenAIAgent requires env so it can serialize the board state.")

        self._step += 1
        prompt = _prompt_for_env(env)

        if self.log_io and self.log_prompt:
            print(
                f"\n========== openai step={self._step} prompt ==========\n"
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
                f"[openai({self.model}) step={self._step} {gen_s:.2f}s] "
                f"raw={text.strip()!r} "
                f"parsed_action={action} "
                f"valid={valid_action} "
                f"chosen={chosen} {decoded} [{tag}]",
                flush=True,
            )

        return int(chosen), None
