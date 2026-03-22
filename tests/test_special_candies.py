from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.candy_env import CandyEnv


def base_board() -> np.ndarray:
    return np.array(
        [[(row + col) % 6 for col in range(8)] for row in range(8)],
        dtype=np.int64,
    )


def make_env() -> CandyEnv:
    env = CandyEnv()
    env.reset(seed=0)
    env.board = base_board()
    env.specials = np.zeros_like(env.board, dtype=np.int8)
    return env


def test_four_match_creates_striped() -> None:
    env = make_env()
    env.board[2, 1:5] = 3
    env._resolve_cascades(preferred_positions=((2, 2),))
    assert np.any(env.specials == env.STRIPED_VERTICAL)


def test_five_match_creates_black() -> None:
    env = make_env()
    env.board[2, 1:6] = 4
    env._resolve_cascades(preferred_positions=((2, 3),))
    assert np.any(env.specials == env.BLACK)


def test_l_or_t_match_creates_wrapped() -> None:
    env = make_env()
    env.board[3, 2:5] = 1
    env.board[2:5, 3] = 1
    env._resolve_cascades(preferred_positions=((3, 3),))
    assert np.any(env.specials == env.WRAPPED)


def test_striped_activation_clears_line() -> None:
    env = make_env()
    env.specials[4, 4] = env.STRIPED_HORIZONTAL
    mask = env._special_effect_mask((4, 4))
    assert mask[4, :].all()
    assert int(mask.sum()) == env.grid_size


def test_wrapped_activation_clears_area() -> None:
    env = make_env()
    env.specials[4, 4] = env.WRAPPED
    mask = env._special_effect_mask((4, 4))
    assert mask[3:6, 3:6].all()
    assert int(mask.sum()) == 9


def test_black_activation_removes_target_type() -> None:
    env = make_env()
    env.specials[0, 0] = env.BLACK
    target_color = 2
    mask = env._special_effect_mask((0, 0), target_color=target_color)
    expected = env.board == target_color
    expected[0, 0] = True
    assert np.array_equal(mask, expected)


def run_all() -> None:
    test_four_match_creates_striped()
    test_five_match_creates_black()
    test_l_or_t_match_creates_wrapped()
    test_striped_activation_clears_line()
    test_wrapped_activation_clears_area()
    test_black_activation_removes_target_type()
    print("special candy tests passed")


if __name__ == "__main__":
    run_all()
