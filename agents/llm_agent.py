from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class MockLLMAgent:
    candidates_min: int = 5
    candidates_max: int = 8

    def propose_actions(self, env) -> list[int]:
        valid = env.valid_actions()
        if not valid:
            return [int(env.action_space.sample())]
        k = random.randint(self.candidates_min, self.candidates_max)
        k = min(k, len(valid))
        return [int(action) for action in random.sample(valid, k)]

    def rank_actions(self, env, actions: list[int]) -> list[tuple[float, int]]:
        ranked = [(float(env.simulate_action_reward(action)), int(action)) for action in actions]
        ranked.sort(reverse=True)
        return ranked

    def predict(self, obs: np.ndarray, env=None, deterministic: bool = True) -> tuple[int, None]:
        if env is None:
            raise ValueError("MockLLMAgent requires env.")
        candidates = self.propose_actions(env)
        ranked = self.rank_actions(env, candidates)
        return int(ranked[0][1]), None
