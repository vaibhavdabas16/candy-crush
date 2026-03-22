from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agents.baselines import GreedyPolicy, RandomPolicy
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import load_ppo
from env.candy_env import CandyEnv


@dataclass
class ViewerConfig:
    cell_size: int = 72
    margin: int = 28
    top_bar: int = 92
    fps: int = 60
    agent_delay: float = 0.35


class CandyViewer:
    COLORS = [
        (235, 72, 78),
        (63, 145, 245),
        (75, 190, 110),
        (248, 214, 76),
        (176, 95, 224),
        (246, 139, 57),
    ]
    BG = (24, 26, 31)
    GRID = (60, 64, 74)
    TEXT = (240, 241, 245)
    MUTED = (174, 179, 190)
    SELECTED = (255, 255, 255)
    MATCH = (255, 255, 255)

    def __init__(
        self,
        env: CandyEnv,
        policy=None,
        mode: str = "manual",
        config: ViewerConfig | None = None,
    ) -> None:
        import pygame

        self.pygame = pygame
        pygame.init()
        self.env = env
        self.policy = policy
        self.mode = mode
        self.config = config or ViewerConfig()
        self.width = self.config.margin * 2 + self.config.cell_size * env.grid_size
        self.height = self.config.top_bar + self.config.margin + self.config.cell_size * env.grid_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Candy Crush RL Viewer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 22)
        self.small_font = pygame.font.SysFont("consolas", 16)
        self.selected: tuple[int, int] | None = None
        self.obs, self.info = self.env.reset(seed=0)
        self.done = False
        self.last_step_time = 0.0

    def run(self) -> None:
        running = True
        while running:
            for event in self.pygame.event.get():
                if event.type == self.pygame.QUIT:
                    running = False
                elif event.type == self.pygame.KEYDOWN:
                    self._handle_key(event.key)
                elif self.mode == "manual" and event.type == self.pygame.MOUSEBUTTONDOWN:
                    self._handle_click(event.pos)

            if self.mode != "manual" and not self.done:
                now = time.time()
                if now - self.last_step_time >= self.config.agent_delay:
                    action = self._agent_action()
                    self._animate_and_step(action)
                    self.last_step_time = now

            self._draw(self.env.board)
            self.pygame.display.flip()
            self.clock.tick(self.config.fps)

        self.pygame.quit()

    def _handle_key(self, key: int) -> None:
        if key == self.pygame.K_ESCAPE:
            self.pygame.event.post(self.pygame.event.Event(self.pygame.QUIT))
        elif key == self.pygame.K_r:
            self.obs, self.info = self.env.reset()
            self.done = False
            self.selected = None
        elif key == self.pygame.K_n and self.mode == "manual" and not self.done:
            valid = self.env.valid_actions()
            action = int(np.random.choice(valid)) if valid else int(self.env.action_space.sample())
            self._animate_and_step(action)

    def _handle_click(self, pos: tuple[int, int]) -> None:
        if self.done:
            return

        cell = self._cell_from_pos(pos)
        if cell is None:
            self.selected = None
            return

        if self.selected is None:
            self.selected = cell
            return

        first = self.selected
        self.selected = None
        if abs(first[0] - cell[0]) + abs(first[1] - cell[1]) != 1:
            self.selected = cell
            return

        action = self.env.encode_action(first, cell)
        self._animate_and_step(action)

    def _agent_action(self) -> int:
        if isinstance(self.policy, (RandomPolicy, GreedyPolicy, DQNAgent)):
            action, _ = self.policy.predict(self.obs, env=self.env, deterministic=True)
            return int(action)

        try:
            action, _ = self.policy.predict(
                self.obs,
                deterministic=True,
                action_masks=self.env.action_masks(),
            )
        except TypeError:
            action, _ = self.policy.predict(self.obs, deterministic=True)
        return int(action)

    def _animate_and_step(self, action: int) -> None:
        if self.done:
            return

        before = self.env.board.copy()
        valid = self.env.is_valid_action(action)
        pos_a, pos_b = self.env.decode_action(action)
        swapped = before.copy()
        swapped[pos_a], swapped[pos_b] = swapped[pos_b], swapped[pos_a]
        match_mask = self.env._find_matches(swapped) if valid else np.zeros_like(swapped, dtype=bool)

        self._animate_swap(before, pos_a, pos_b)
        if valid and match_mask.any():
            self._animate_clear(swapped, match_mask)

        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.done = terminated or truncated
        self._animate_fall(swapped, self.env.board.copy())

    def _animate_swap(
        self,
        board: np.ndarray,
        pos_a: tuple[int, int],
        pos_b: tuple[int, int],
        frames: int = 14,
    ) -> None:
        for frame in range(frames + 1):
            t = frame / frames
            offsets = {
                pos_a: (
                    (pos_b[1] - pos_a[1]) * self.config.cell_size * t,
                    (pos_b[0] - pos_a[0]) * self.config.cell_size * t,
                ),
                pos_b: (
                    (pos_a[1] - pos_b[1]) * self.config.cell_size * t,
                    (pos_a[0] - pos_b[0]) * self.config.cell_size * t,
                ),
            }
            self._draw(board, offsets=offsets)
            self.pygame.display.flip()
            self.clock.tick(self.config.fps)

    def _animate_clear(self, board: np.ndarray, mask: np.ndarray, frames: int = 12) -> None:
        for frame in range(frames + 1):
            alpha = 1.0 - frame / frames
            self._draw(board, clear_mask=mask, clear_alpha=alpha)
            self.pygame.display.flip()
            self.clock.tick(self.config.fps)

    def _animate_fall(self, start: np.ndarray, end: np.ndarray, frames: int = 18) -> None:
        for frame in range(frames + 1):
            t = frame / frames
            if t < 0.5:
                self._draw(start)
            else:
                drop = int((1.0 - t) * self.config.cell_size * 2)
                offsets = {
                    (r, c): (0, -drop)
                    for r in range(self.env.grid_size)
                    for c in range(self.env.grid_size)
                }
                self._draw(end, offsets=offsets)
            self.pygame.display.flip()
            self.clock.tick(self.config.fps)

    def _draw(
        self,
        board: np.ndarray,
        offsets: dict[tuple[int, int], tuple[float, float]] | None = None,
        clear_mask: np.ndarray | None = None,
        clear_alpha: float = 1.0,
    ) -> None:
        offsets = offsets or {}
        self.screen.fill(self.BG)
        self._draw_header()

        for row in range(self.env.grid_size):
            for col in range(self.env.grid_size):
                candy = int(board[row, col])
                if candy < 0:
                    continue
                ox, oy = offsets.get((row, col), (0, 0))
                rect = self._cell_rect(row, col).move(ox, oy)
                self.pygame.draw.rect(self.screen, self.GRID, rect, border_radius=8)
                inset = 9
                candy_rect = rect.inflate(-inset * 2, -inset * 2)
                color = self.COLORS[candy % len(self.COLORS)]
                self.pygame.draw.ellipse(self.screen, color, candy_rect)
                self.pygame.draw.ellipse(self.screen, (255, 255, 255), candy_rect, width=2)
                self._draw_special_marker((row, col), candy_rect)

                if clear_mask is not None and clear_mask[row, col]:
                    overlay = self.pygame.Surface((rect.width, rect.height), self.pygame.SRCALPHA)
                    overlay_alpha = int(190 * clear_alpha)
                    overlay.fill((*self.MATCH, overlay_alpha))
                    self.screen.blit(overlay, rect)

        if self.selected is not None:
            self.pygame.draw.rect(
                self.screen,
                self.SELECTED,
                self._cell_rect(*self.selected),
                width=4,
                border_radius=8,
            )

    def _draw_header(self) -> None:
        status = "done" if self.done else self.mode
        title = f"score={self.env.score:.1f} moves={self.env.moves_left} mode={status}"
        help_text = "manual: click adjacent candies | N random step | R reset | Esc quit"
        if self.mode != "manual":
            help_text = "agent mode | R reset | Esc quit"
        self.screen.blit(self.font.render(title, True, self.TEXT), (self.config.margin, 24))
        self.screen.blit(self.small_font.render(help_text, True, self.MUTED), (self.config.margin, 58))

    def _draw_special_marker(self, pos: tuple[int, int], rect) -> None:
        special = int(self.env.specials[pos])
        if special == self.env.NORMAL:
            return

        cx, cy = rect.center
        if special == self.env.STRIPED_HORIZONTAL:
            self.pygame.draw.line(self.screen, (255, 255, 255), (rect.left, cy), (rect.right, cy), 4)
        elif special == self.env.STRIPED_VERTICAL:
            self.pygame.draw.line(self.screen, (255, 255, 255), (cx, rect.top), (cx, rect.bottom), 4)
        elif special == self.env.WRAPPED:
            wrapped_rect = rect.inflate(-12, -12)
            self.pygame.draw.rect(self.screen, (255, 255, 255), wrapped_rect, width=4, border_radius=6)
        elif special == self.env.BLACK:
            self.pygame.draw.circle(self.screen, (20, 20, 24), rect.center, rect.width // 3)
            self.pygame.draw.circle(self.screen, (255, 255, 255), rect.center, rect.width // 3, 3)

    def _cell_rect(self, row: int, col: int):
        x = self.config.margin + col * self.config.cell_size
        y = self.config.top_bar + row * self.config.cell_size
        return self.pygame.Rect(x, y, self.config.cell_size, self.config.cell_size)

    def _cell_from_pos(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        x, y = pos
        col = (x - self.config.margin) // self.config.cell_size
        row = (y - self.config.top_bar) // self.config.cell_size
        if 0 <= row < self.env.grid_size and 0 <= col < self.env.grid_size:
            return int(row), int(col)
        return None


def load_policy(agent_name: str, dqn_path: str | Path, ppo_path: str | Path, env: CandyEnv):
    agent_name = agent_name.lower()
    if agent_name == "manual":
        return None
    if agent_name == "random":
        return RandomPolicy()
    if agent_name == "greedy":
        return GreedyPolicy()
    if agent_name == "dqn":
        path = Path(dqn_path)
        if not path.exists():
            raise FileNotFoundError(f"DQN model not found: {path}")
        return DQNAgent.load(path)
    if agent_name == "ppo":
        path = Path(ppo_path)
        if not path.exists() and not Path(str(path) + ".zip").exists():
            raise FileNotFoundError(f"PPO model not found: {path}.zip")
        return load_ppo(path, env=env)
    raise ValueError(f"Unknown agent: {agent_name}")
