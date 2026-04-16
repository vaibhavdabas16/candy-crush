from __future__ import annotations

import random

import numpy as np


class RandomPolicy:
    def predict(self, obs: np.ndarray, env=None, deterministic: bool = True) -> tuple[int, None]:
        if env is None:
            raise ValueError("RandomPolicy requires env to choose from valid actions.")
        valid = env.valid_actions()
        if not valid:
            return int(env.action_space.sample()), None
        return int(random.choice(valid)), None


class GreedyPolicy:
    def predict(self, obs: np.ndarray, env=None, deterministic: bool = True) -> tuple[int, None]:
        if env is None:
            raise ValueError("GreedyPolicy requires env to simulate actions.")
        valid = env.valid_actions()
        if not valid:
            return int(env.action_space.sample()), None

        rewards = [(env.simulate_action_reward(action), action) for action in valid]
        best_reward = max(reward for reward, _ in rewards)
        best_actions = [action for reward, action in rewards if reward == best_reward]
        return int(random.choice(best_actions)), None
