from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from env.candy_env import CandyEnv


@dataclass(frozen=True)
class SpecialInjectionConfig:
    min_specials: int = 1
    max_specials: int = 3
    enabled_specials: tuple[int, ...] = (
        CandyEnv.STRIPED_HORIZONTAL,
        CandyEnv.STRIPED_VERTICAL,
        CandyEnv.WRAPPED,
        CandyEnv.BLACK,
    )


def inject_random_specials(
    env: CandyEnv,
    seed: int,
    config: SpecialInjectionConfig | None = None,
) -> None:
    """Add random special candies to an already-reset board for stress testing."""
    config = config or SpecialInjectionConfig()
    enabled = tuple(int(v) for v in config.enabled_specials if int(v) != CandyEnv.NORMAL)
    if not enabled or config.max_specials <= 0:
        return

    rng = np.random.default_rng(seed)
    total_cells = env.grid_size * env.grid_size
    count = int(rng.integers(config.min_specials, config.max_specials + 1))
    count = max(0, min(count, total_cells))
    cells = rng.choice(total_cells, size=count, replace=False)
    for flat_idx in cells:
        row = int(flat_idx) // env.grid_size
        col = int(flat_idx) % env.grid_size
        env.specials[row, col] = int(rng.choice(enabled))


def reset_with_optional_specials(
    seed: int,
    max_moves: int,
    special_seed: int | None = None,
    special_config: SpecialInjectionConfig | None = None,
) -> CandyEnv:
    env = CandyEnv(max_moves=max_moves)
    env.reset(seed=seed)
    if special_seed is not None:
        inject_random_specials(env, special_seed, special_config)
    return env
