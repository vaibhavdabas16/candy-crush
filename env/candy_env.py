from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CandyEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}
    NORMAL = 0
    STRIPED_HORIZONTAL = 1
    STRIPED_VERTICAL = 2
    WRAPPED = 3
    BLACK = 4
    SPECIAL_BONUSES = {
        STRIPED_HORIZONTAL: 20.0,
        STRIPED_VERTICAL: 20.0,
        WRAPPED: 40.0,
        BLACK: 60.0,
    }

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
        self.specials = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.moves_left = self.max_moves
        self.score = 0.0
        self.last_action: int | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.moves_left = self.max_moves
        self.score = 0.0
        self.last_action = None
        self.board = self._generate_start_board()
        self.specials = np.zeros_like(self.board, dtype=np.int8)
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        invalid = False
        self.last_action = action

        if action < 0 or action >= self.num_actions or not self.is_valid_action(action):
            reward = self.invalid_penalty
            invalid = True
        else:
            pos_a, pos_b = self.decode_action(action)
            special_targets = self._special_targets_before_swap(pos_a, pos_b)
            self._swap(pos_a, pos_b)
            reward = float(
                self._resolve_cascades(
                    initial_special_positions=(pos_a, pos_b),
                    special_targets=special_targets,
                    preferred_positions=(pos_a, pos_b),
                )
            )

        self.moves_left -= 1
        self.score += reward
        terminated = False
        truncated = self.moves_left <= 0

        info = self._get_info()
        info["invalid"] = invalid
        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode: str | None = None) -> str | None:
        render_mode = mode or self.render_mode
        last_action = "None"
        if self.last_action is not None and 0 <= self.last_action < self.num_actions:
            pos_a, pos_b = self.decode_action(self.last_action)
            last_action = f"{self.last_action}: {pos_a} <-> {pos_b}"
        text = (
            f"Score: {self.score:.1f} | Moves left: {self.moves_left} | Last action: {last_action}\n"
            + "\n".join(
                " ".join(self._render_cell((row, col)) for col in range(self.grid_size))
                for row in range(self.grid_size)
            )
        )
        if render_mode == "human":
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
        if self._is_special(pos_a) or self._is_special(pos_b):
            return True

        if self.board[pos_a] == self.board[pos_b]:
            return False

        board = self.board.copy()
        board[pos_a], board[pos_b] = board[pos_b], board[pos_a]
        return bool(self._find_matches(board).any())

    def valid_actions(self) -> list[int]:
        mask = self.get_action_mask()
        return [int(a) for a in np.flatnonzero(mask)]

    def get_action_mask(self) -> np.ndarray:
        """Return a binary vector where 1 marks swaps that create a match."""
        mask = self._compute_valid_action_mask()
        if not mask.any():
            mask[:] = 1
        return mask

    def _compute_valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=np.int8)
        for action in range(self.num_actions):
            if self.is_valid_action(action):
                mask[action] = 1
        return mask

    def action_masks(self) -> np.ndarray:
        return self.get_action_mask().astype(bool)

    def simulate_action_reward(self, action: int) -> float:
        board = self.board.copy()
        specials = self.specials.copy()
        rng_state = deepcopy(self.np_random.bit_generator.state)
        moves_left = self.moves_left
        score = self.score

        try:
            if not self.is_valid_action(action):
                return float(self.invalid_penalty)
            pos_a, pos_b = self.decode_action(action)
            special_targets = self._special_targets_before_swap(pos_a, pos_b)
            self._swap(pos_a, pos_b)
            return float(
                self._resolve_cascades(
                    initial_special_positions=(pos_a, pos_b),
                    special_targets=special_targets,
                    preferred_positions=(pos_a, pos_b),
                )
            )
        finally:
            self.board = board
            self.specials = specials
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
        cloned.specials = self.specials.copy()
        cloned.moves_left = self.moves_left
        cloned.score = self.score
        cloned.last_action = self.last_action
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
            "valid_action_mask": self.get_action_mask(),
            "last_action": self.last_action,
            "specials": self.specials.copy(),
        }

    def _generate_start_board(self) -> np.ndarray:
        for _ in range(10_000):
            board = self.np_random.integers(
                0,
                self.candy_types,
                size=(self.grid_size, self.grid_size),
                dtype=np.int64,
            )
            self.specials = np.zeros_like(board, dtype=np.int8)
            while self._find_matches(board).any():
                mask = self._find_matches(board)
                board[mask] = self.np_random.integers(0, self.candy_types, size=int(mask.sum()))
            self.board = board
            if self._compute_valid_action_mask().any():
                return board
        raise RuntimeError("Could not generate a playable board.")

    def _swap(self, pos_a: tuple[int, int], pos_b: tuple[int, int]) -> None:
        self.board[pos_a], self.board[pos_b] = self.board[pos_b], self.board[pos_a]
        self.specials[pos_a], self.specials[pos_b] = self.specials[pos_b], self.specials[pos_a]

    def _resolve_cascades(
        self,
        initial_special_positions: tuple[tuple[int, int], ...] = (),
        special_targets: dict[tuple[int, int], int] | None = None,
        preferred_positions: tuple[tuple[int, int], ...] = (),
    ) -> float:
        total_reward = 0.0
        special_targets = special_targets or {}

        if initial_special_positions:
            clear_mask, special_bonus = self._expand_special_effects(
                self._mask_from_positions(initial_special_positions),
                special_targets,
            )
            removed = int(clear_mask.sum())
            if removed:
                total_reward += float(removed**2) + special_bonus
                self.board[clear_mask] = -1
                self.specials[clear_mask] = self.NORMAL
                self._apply_gravity_and_fill()

        while True:
            groups = self._find_match_groups(self.board)
            matches = self._groups_to_mask(groups)
            removed = int(matches.sum())
            if removed == 0:
                break

            creations = self._plan_special_creations(groups, preferred_positions)
            clear_mask, special_bonus = self._expand_special_effects(matches)
            for pos in creations:
                clear_mask[pos] = False

            removed = int(clear_mask.sum())
            if removed == 0:
                break

            total_reward += float(removed**2 + 10) + special_bonus
            self.board[clear_mask] = -1
            self.specials[clear_mask] = self.NORMAL
            for pos, special_type in creations.items():
                self.specials[pos] = special_type
            self._apply_gravity_and_fill()
            preferred_positions = ()

        return total_reward

    def _special_targets_before_swap(
        self,
        pos_a: tuple[int, int],
        pos_b: tuple[int, int],
    ) -> dict[tuple[int, int], int]:
        targets: dict[tuple[int, int], int] = {}
        if self.specials[pos_a] == self.BLACK:
            targets[pos_b] = int(self.board[pos_b])
        if self.specials[pos_b] == self.BLACK:
            targets[pos_a] = int(self.board[pos_a])
        return targets

    def _is_special(self, pos: tuple[int, int]) -> bool:
        return bool(self.specials[pos] != self.NORMAL)

    def _mask_from_positions(self, positions: tuple[tuple[int, int], ...]) -> np.ndarray:
        mask = np.zeros_like(self.board, dtype=bool)
        for row, col in positions:
            if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
                if self.specials[row, col] != self.NORMAL:
                    mask[row, col] = True
        return mask

    def _expand_special_effects(
        self,
        clear_mask: np.ndarray,
        special_targets: dict[tuple[int, int], int] | None = None,
    ) -> tuple[np.ndarray, float]:
        special_targets = special_targets or {}
        expanded = clear_mask.copy()
        processed: set[tuple[int, int]] = set()
        bonus = 0.0

        while True:
            special_positions = [
                (int(row), int(col))
                for row, col in np.argwhere(expanded & (self.specials != self.NORMAL))
                if (int(row), int(col)) not in processed
            ]
            if not special_positions:
                break

            for pos in special_positions:
                processed.add(pos)
                special_type = int(self.specials[pos])
                target_color = special_targets.get(pos)
                expanded |= self._special_effect_mask(pos, target_color)
                bonus += self.SPECIAL_BONUSES.get(special_type, 0.0)

        return expanded, bonus

    def _special_effect_mask(
        self,
        pos: tuple[int, int],
        target_color: int | None = None,
    ) -> np.ndarray:
        row, col = pos
        special_type = int(self.specials[pos])
        mask = np.zeros_like(self.board, dtype=bool)

        if special_type == self.STRIPED_HORIZONTAL:
            mask[row, :] = self.board[row, :] >= 0
        elif special_type == self.STRIPED_VERTICAL:
            mask[:, col] = self.board[:, col] >= 0
        elif special_type == self.WRAPPED:
            r0 = max(0, row - 1)
            r1 = min(self.grid_size, row + 2)
            c0 = max(0, col - 1)
            c1 = min(self.grid_size, col + 2)
            mask[r0:r1, c0:c1] = self.board[r0:r1, c0:c1] >= 0
        elif special_type == self.BLACK:
            if target_color is None or target_color < 0:
                target_color = int(self.board[pos])
            mask = self.board == target_color
            mask[pos] = True
        else:
            mask[pos] = True

        return mask

    def _find_match_groups(self, board: np.ndarray) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []

        for row in range(self.grid_size):
            start = 0
            while start < self.grid_size:
                value = board[row, start]
                end = start + 1
                while end < self.grid_size and board[row, end] == value:
                    end += 1
                if value >= 0 and end - start >= 3:
                    groups.append(
                        {
                            "orientation": "horizontal",
                            "cells": [(row, col) for col in range(start, end)],
                        }
                    )
                start = end

        for col in range(self.grid_size):
            start = 0
            while start < self.grid_size:
                value = board[start, col]
                end = start + 1
                while end < self.grid_size and board[end, col] == value:
                    end += 1
                if value >= 0 and end - start >= 3:
                    groups.append(
                        {
                            "orientation": "vertical",
                            "cells": [(row, col) for row in range(start, end)],
                        }
                    )
                start = end

        return groups

    def _groups_to_mask(self, groups: list[dict[str, Any]]) -> np.ndarray:
        mask = np.zeros_like(self.board, dtype=bool)
        for group in groups:
            for pos in group["cells"]:
                mask[pos] = True
        return mask

    def _plan_special_creations(
        self,
        groups: list[dict[str, Any]],
        preferred_positions: tuple[tuple[int, int], ...] = (),
    ) -> dict[tuple[int, int], int]:
        creations: dict[tuple[int, int], int] = {}
        used_groups: set[int] = set()

        for h_idx, h_group in enumerate(groups):
            if h_group["orientation"] != "horizontal":
                continue
            h_cells = set(h_group["cells"])
            for v_idx, v_group in enumerate(groups):
                if v_group["orientation"] != "vertical":
                    continue
                overlap = h_cells & set(v_group["cells"])
                if overlap:
                    union = h_cells | set(v_group["cells"])
                    pos = self._choose_special_position(union, preferred_positions)
                    creations[pos] = self.WRAPPED
                    used_groups.update({h_idx, v_idx})

        for idx, group in enumerate(groups):
            if idx in used_groups:
                continue
            cells = group["cells"]
            if len(cells) >= 5:
                pos = self._choose_special_position(cells, preferred_positions)
                creations[pos] = self.BLACK
            elif len(cells) == 4:
                pos = self._choose_special_position(cells, preferred_positions)
                if group["orientation"] == "horizontal":
                    creations[pos] = self.STRIPED_VERTICAL
                else:
                    creations[pos] = self.STRIPED_HORIZONTAL

        return creations

    def _choose_special_position(
        self,
        cells: set[tuple[int, int]] | list[tuple[int, int]],
        preferred_positions: tuple[tuple[int, int], ...],
    ) -> tuple[int, int]:
        cell_list = list(cells)
        for pos in preferred_positions:
            if pos in cell_list:
                return pos
        cell_list.sort()
        return cell_list[len(cell_list) // 2]

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
            special_column = self.specials[:, col]
            keep = column >= 0
            remaining = column[keep]
            remaining_specials = special_column[keep]
            missing = self.grid_size - len(remaining)
            new_values = self.np_random.integers(0, self.candy_types, size=missing, dtype=np.int64)
            new_specials = np.zeros(missing, dtype=np.int8)
            self.board[:, col] = np.concatenate([new_values, remaining])
            self.specials[:, col] = np.concatenate([new_specials, remaining_specials])

    def _render_cell(self, pos: tuple[int, int]) -> str:
        value = int(self.board[pos])
        special = int(self.specials[pos])
        if value < 0:
            return ".."
        if special == self.STRIPED_HORIZONTAL:
            return f"{value}H"
        if special == self.STRIPED_VERTICAL:
            return f"{value}V"
        if special == self.WRAPPED:
            return f"{value}W"
        if special == self.BLACK:
            return "B*"
        return f"{value}."
