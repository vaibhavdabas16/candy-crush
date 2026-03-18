from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CandyEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        grid_size: int = 8,
        candy_types: int = 6,
        max_moves: int = 20,
        invalid_penalty: float = -5.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.candy_types = candy_types
        self.max_moves = max_moves
        self.invalid_penalty = invalid_penalty
        self.render_mode = render_mode

        self.num_horizontal_actions = self.grid_size * (self.grid_size - 1)
        self.num_vertical_actions = (self.grid_size - 1) * self.grid_size
        self.num_actions = self.num_horizontal_actions + self.num_vertical_actions

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.grid_size * self.grid_size + 1,),
            dtype=np.float32,
        )

        self.board = np.zeros((self.grid_size, self.grid_size), dtype=np.int64)
        self.moves_left = self.max_moves
        self.score = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.moves_left = self.max_moves
        self.score = 0.0
        self.board = self._generate_start_board()
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        invalid = False

        if action < 0 or action >= self.num_actions or not self.is_valid_action(action):
            reward = self.invalid_penalty
            invalid = True
        else:
            pos_a, pos_b = self.decode_action(action)
            self._swap(pos_a, pos_b)
            reward = float(self._resolve_cascades())

        self.moves_left -= 1
        self.score += reward
        terminated = False
        truncated = self.moves_left <= 0

        info = self._get_info()
        info["invalid"] = invalid
        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> str | None:
        text = (
            f"Score: {self.score:.1f} | Moves left: {self.moves_left}\n"
            + "\n".join(" ".join(str(int(v)) for v in row) for row in self.board)
        )
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def decode_action(self, action: int) -> tuple[tuple[int, int], tuple[int, int]]:
        if action < self.num_horizontal_actions:
            row = action // (self.grid_size - 1)
            col = action % (self.grid_size - 1)
            return (row, col), (row, col + 1)

        vertical_action = action - self.num_horizontal_actions
        row = vertical_action // self.grid_size
        col = vertical_action % self.grid_size
        return (row, col), (row + 1, col)

    def encode_action(self, pos_a: tuple[int, int], pos_b: tuple[int, int]) -> int:
        (r1, c1), (r2, c2) = pos_a, pos_b
        if r1 == r2 and abs(c1 - c2) == 1:
            col = min(c1, c2)
            return r1 * (self.grid_size - 1) + col
        if c1 == c2 and abs(r1 - r2) == 1:
            row = min(r1, r2)
            return self.num_horizontal_actions + row * self.grid_size + c1
        raise ValueError(f"Positions {pos_a} and {pos_b} are not adjacent.")

    def is_valid_action(self, action: int) -> bool:
        if action < 0 or action >= self.num_actions:
            return False

        pos_a, pos_b = self.decode_action(action)
        if self.board[pos_a] == self.board[pos_b]:
            return False

        board = self.board.copy()
        board[pos_a], board[pos_b] = board[pos_b], board[pos_a]
        return bool(self._find_matches(board).any())

    def valid_actions(self) -> list[int]:
        return [a for a in range(self.num_actions) if self.is_valid_action(a)]

    def action_masks(self) -> np.ndarray:
        return np.array([self.is_valid_action(a) for a in range(self.num_actions)], dtype=bool)

    def simulate_action_reward(self, action: int) -> float:
        board = self.board.copy()
        rng_state = deepcopy(self.np_random.bit_generator.state)
        moves_left = self.moves_left
        score = self.score

        try:
            if not self.is_valid_action(action):
                return float(self.invalid_penalty)
            pos_a, pos_b = self.decode_action(action)
            self._swap(pos_a, pos_b)
            return float(self._resolve_cascades())
        finally:
            self.board = board
            self.np_random.bit_generator.state = rng_state
            self.moves_left = moves_left
            self.score = score

    def clone(self) -> "CandyEnv":
        cloned = CandyEnv(
            grid_size=self.grid_size,
            candy_types=self.candy_types,
            max_moves=self.max_moves,
            invalid_penalty=self.invalid_penalty,
            render_mode=self.render_mode,
        )
        cloned.board = self.board.copy()
        cloned.moves_left = self.moves_left
        cloned.score = self.score
        cloned.np_random.bit_generator.state = deepcopy(self.np_random.bit_generator.state)
        return cloned

    def _get_obs(self) -> np.ndarray:
        board_obs = self.board.astype(np.float32).reshape(-1) / float(self.candy_types - 1)
        moves_obs = np.array([self.moves_left / float(self.max_moves)], dtype=np.float32)
        return np.concatenate([board_obs, moves_obs]).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        valid = self.valid_actions()
        return {
            "score": self.score,
            "moves_left": self.moves_left,
            "valid_actions": valid,
            "valid_action_mask": self.action_masks(),
        }

    def _generate_start_board(self) -> np.ndarray:
        for _ in range(10_000):
            board = self.np_random.integers(
                0,
                self.candy_types,
                size=(self.grid_size, self.grid_size),
                dtype=np.int64,
            )
            while self._find_matches(board).any():
                mask = self._find_matches(board)
                board[mask] = self.np_random.integers(0, self.candy_types, size=int(mask.sum()))
            self.board = board
            if self.valid_actions():
                return board
        raise RuntimeError("Could not generate a playable board.")

    def _swap(self, pos_a: tuple[int, int], pos_b: tuple[int, int]) -> None:
        self.board[pos_a], self.board[pos_b] = self.board[pos_b], self.board[pos_a]

    def _resolve_cascades(self) -> float:
        total_reward = 0.0
        while True:
            matches = self._find_matches(self.board)
            removed = int(matches.sum())
            if removed == 0:
                break

            total_reward += float(removed**2 + 10)
            self.board[matches] = -1
            self._apply_gravity_and_fill()

        return total_reward

    def _find_matches(self, board: np.ndarray) -> np.ndarray:
        matches = np.zeros_like(board, dtype=bool)

        for row in range(self.grid_size):
            start = 0
            while start < self.grid_size:
                value = board[row, start]
                end = start + 1
                while end < self.grid_size and board[row, end] == value:
                    end += 1
                if value >= 0 and end - start >= 3:
                    matches[row, start:end] = True
                start = end

        for col in range(self.grid_size):
            start = 0
            while start < self.grid_size:
                value = board[start, col]
                end = start + 1
                while end < self.grid_size and board[end, col] == value:
                    end += 1
                if value >= 0 and end - start >= 3:
                    matches[start:end, col] = True
                start = end

        return matches

    def _apply_gravity_and_fill(self) -> None:
        for col in range(self.grid_size):
            column = self.board[:, col]
            remaining = column[column >= 0]
            missing = self.grid_size - len(remaining)
            new_values = self.np_random.integers(0, self.candy_types, size=missing, dtype=np.int64)
            self.board[:, col] = np.concatenate([new_values, remaining])
