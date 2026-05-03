from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from env.candy_env import CandyEnv
from train.train_llm_grpo_candy import action_from_text, generate_swap, load_lora_model, load_tokenizer


class LLMGRPOAgent:
    """Qwen LoRA text policy for Candy Crush swap recommendations."""

    def __init__(
        self,
        adapter_path: str | Path,
        model_name: str = "Qwen/Qwen3.5-9B",
        lora_rank: int = 64,
        use_4bit: bool = True,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> None:
        self.adapter_path = Path(adapter_path)
        self.model_name = model_name
        self.lora_rank = lora_rank
        self.use_4bit = use_4bit
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = load_tokenizer(model_name)
        self.model = load_lora_model(
            model_name=model_name,
            adapter_path=str(self.adapter_path),
            lora_rank=lora_rank,
            use_4bit=use_4bit,
            beta_kl=0.0,
            device=device,
            dtype=dtype,
            is_trainable=False,
        )
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True

    def predict(self, obs: np.ndarray, env: CandyEnv | None = None, deterministic: bool = True) -> tuple[int, None]:
        if env is None:
            raise ValueError("LLMGRPOAgent requires env so it can serialize the board state.")

        text = generate_swap(
            self.model,
            self.tokenizer,
            env,
            max_new_tokens=self.max_new_tokens,
            temperature=0.0 if deterministic else self.temperature,
        )
        action = action_from_text(env, text)
        if action is not None and env.is_valid_action(action):
            return int(action), None

        valid = env.valid_actions()
        if valid:
            action = max(valid, key=lambda a: env.simulate_action_reward(int(a)))
            return int(action), None
        return int(env.action_space.sample()), None

    def close(self) -> None:
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
