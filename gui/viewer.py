from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agents.baselines import GreedyPolicy, RandomPolicy
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import load_ppo
from env.candy_env import CandyEnv

ASSETS_DIR = Path(__file__).resolve().parent
SPRITE_PATH = ASSETS_DIR / "candy_sprites.png"
BOARD_BG_PATH = ASSETS_DIR / "board_bg.png"

SPRITE_CENTERS = [
    (140, 140),
    (320, 140),
    (500, 140),
    (140, 440),
    (320, 440),
    (500, 440),
]
SPRITE_CROP_RADIUS = 95


@dataclass
class ViewerConfig:
    cell_size: int = 80
    margin: int = 32
    top_bar: int = 100
    fps: int = 60
    agent_delay: float = 0.35


class CandyViewer:
    COLORS = [
        (235, 72, 78),
        (75, 190, 110),
        (63, 145, 245),
        (248, 214, 76),
        (176, 95, 224),
        (246, 139, 57),
    ]
    BG = (18, 20, 28)
    PANEL = (30, 34, 46)
    GRID = (48, 54, 70)
    GRID_HI = (70, 78, 98)
    TEXT = (240, 241, 245)
    MUTED = (174, 179, 190)
    ACCENT = (255, 215, 96)
    SELECTED = (255, 230, 120)
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
        self.candy_sprites = self._load_candy_sprites()
        self.board_bg = self._load_board_bg()

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
        self._animate_fall(swapped, self.env.board.copy(), match_mask)

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

    def _animate_fall(
        self,
        start: np.ndarray,
        end: np.ndarray,
        clear_mask: np.ndarray | None = None,
        frames: int = 18,
    ) -> None:
        grid = self.env.grid_size
        cell = self.config.cell_size

        if clear_mask is None or not clear_mask.any():
            self._draw(end)
            self.pygame.display.flip()
            self.clock.tick(self.config.fps)
            return

        # Per-cell starting y-offset (pixels). Only cells that moved get an entry;
        # unchanged cells stay stationary (no offset => no visual refresh).
        move_offsets: dict[tuple[int, int], int] = {}
        for c in range(grid):
            cleared_col = clear_mask[:, c]
            survivors = [r for r in range(grid) if not cleared_col[r]]
            num_new = grid - len(survivors)
            # Survivors land at bottom of column in same relative order.
            for i, orig_r in enumerate(survivors):
                final_r = num_new + i
                if final_r != orig_r:
                    move_offsets[(final_r, c)] = (orig_r - final_r) * cell  # negative y offset (starts higher)
            # Newly-spawned cells fill top rows; start above the board.
            for i in range(num_new):
                move_offsets[(i, c)] = -(num_new - i) * cell

        if not move_offsets:
            self._draw(end)
            self.pygame.display.flip()
            self.clock.tick(self.config.fps)
            return

        for frame in range(frames + 1):
            t = frame / frames
            eased = t * t * (3 - 2 * t)  # smoothstep
            offsets = {
                pos: (0, total_y * (1 - eased))
                for pos, total_y in move_offsets.items()
            }
            self._draw(end, offsets=offsets)
            self.pygame.display.flip()
            self.clock.tick(self.config.fps)

    def _load_candy_sprites(self) -> list:
        pygame = self.pygame
        sprites: list = []
        cell = self.config.cell_size
        target = cell - 12
        if not SPRITE_PATH.exists():
            return []
        sheet = pygame.image.load(str(SPRITE_PATH)).convert_alpha()
        sheet = self._strip_white_background(sheet)
        sheet_w, sheet_h = sheet.get_size()
        scale_x = sheet_w / 640.0
        scale_y = sheet_h / 640.0
        for cx, cy in SPRITE_CENTERS:
            r = int(SPRITE_CROP_RADIUS * scale_x)
            x = int(cx * scale_x) - r
            y = int(cy * scale_y) - r
            w = h = r * 2
            x = max(0, min(x, sheet_w - w))
            y = max(0, min(y, sheet_h - h))
            sub = sheet.subsurface(pygame.Rect(x, y, w, h)).copy()
            scaled = pygame.transform.smoothscale(sub, (target, target))
            sprites.append(scaled)
        return sprites

    def _strip_white_background(self, surface) -> "object":
        """Remove the sprite sheet's white background plus its gray shadow halo.

        The sheet has saturated-color candies on a near-white field with soft
        gray drop shadows. Treat any low-saturation light pixel as background.
        """
        pygame = self.pygame
        surface = surface.convert_alpha()
        arr = pygame.surfarray.pixels_alpha(surface)
        rgb = pygame.surfarray.pixels3d(surface).astype(np.int16)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        saturation = max_c - min_c

        background = (max_c > 190) & (saturation < 45)
        arr[background] = 0

        edge = (~background) & (max_c > 170) & (saturation < 70)
        arr[edge] = (arr[edge].astype(np.int16) * 40 // 255).astype(arr.dtype)
        del arr, rgb
        return surface

    def _load_board_bg(self):
        return None

    def _draw(
        self,
        board: np.ndarray,
        offsets: dict[tuple[int, int], tuple[float, float]] | None = None,
        clear_mask: np.ndarray | None = None,
        clear_alpha: float = 1.0,
    ) -> None:
        offsets = offsets or {}
        pygame = self.pygame
        self.screen.fill(self.BG)
        self._draw_header()

        grid_size = self.env.grid_size
        board_w = self.config.cell_size * grid_size
        board_rect = pygame.Rect(
            self.config.margin, self.config.top_bar, board_w, board_w
        )
        frame = board_rect.inflate(16, 16)
        pygame.draw.rect(self.screen, self.PANEL, frame, border_radius=16)
        pygame.draw.rect(self.screen, self.GRID_HI, frame, width=2, border_radius=16)

        inner_pad = 4
        inner_rect = board_rect.inflate(-inner_pad * 2, -inner_pad * 2)
        pygame.draw.rect(self.screen, (22, 26, 38), inner_rect, border_radius=10)

        tile_inset = 3
        for row in range(grid_size):
            for col in range(grid_size):
                tile = self._cell_rect(row, col).inflate(-tile_inset * 2, -tile_inset * 2)
                shade = self.GRID if (row + col) % 2 == 0 else self.GRID_HI
                pygame.draw.rect(self.screen, shade, tile, border_radius=8)

        prev_clip = self.screen.get_clip()
        self.screen.set_clip(inner_rect)
        for row in range(grid_size):
            for col in range(grid_size):
                candy = int(board[row, col])
                if candy < 0:
                    continue
                ox, oy = offsets.get((row, col), (0, 0))
                rect = self._cell_rect(row, col).move(ox, oy)
                candy_rect = rect.inflate(-6, -6)
                self._draw_candy(candy, candy_rect)
                self._draw_special_marker((row, col), candy_rect)

                if clear_mask is not None and clear_mask[row, col]:
                    overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                    overlay_alpha = int(200 * clear_alpha)
                    overlay.fill((255, 255, 210, overlay_alpha))
                    self.screen.blit(overlay, rect)
        self.screen.set_clip(prev_clip)

        if self.selected is not None:
            sel_rect = self._cell_rect(*self.selected).inflate(4, 4)
            pygame.draw.rect(
                self.screen,
                self.SELECTED,
                sel_rect,
                width=4,
                border_radius=10,
            )

    def _draw_candy(self, candy: int, rect) -> None:
        pygame = self.pygame
        if self.candy_sprites:
            sprite = self.candy_sprites[candy % len(self.candy_sprites)]
            sprite_rect = sprite.get_rect(center=rect.center)
            shadow = pygame.Surface(sprite.get_size(), pygame.SRCALPHA)
            shadow.blit(sprite, (0, 0))
            shadow.fill((0, 0, 0, 110), special_flags=pygame.BLEND_RGBA_MULT)
            self.screen.blit(shadow, sprite_rect.move(2, 3))
            self.screen.blit(sprite, sprite_rect)
            return
        color = self.COLORS[candy % len(self.COLORS)]
        pygame.draw.ellipse(self.screen, color, rect)
        pygame.draw.ellipse(self.screen, (255, 255, 255), rect, width=2)

    def _draw_header(self) -> None:
        pygame = self.pygame
        board_w = self.config.cell_size * self.env.grid_size
        header_rect = pygame.Rect(
            self.config.margin - 8,
            16,
            board_w + 16,
            self.config.top_bar - 28,
        )
        pygame.draw.rect(self.screen, self.PANEL, header_rect, border_radius=12)
        pygame.draw.rect(self.screen, self.GRID_HI, header_rect, width=2, border_radius=12)

        status = "done" if self.done else self.mode
        score_text = f"Score: {self.env.score:.1f}"
        moves_text = f"Moves left: {self.env.moves_left}"
        mode_text = f"Mode: {status}"
        help_text = "Click adjacent candies | N random | R reset | Esc quit"
        if self.mode != "manual":
            help_text = f"Agent: {self.mode} | R reset | Esc quit"

        self.screen.blit(
            self.font.render(score_text, True, self.ACCENT),
            (header_rect.left + 14, header_rect.top + 8),
        )
        self.screen.blit(
            self.font.render(moves_text, True, self.TEXT),
            (header_rect.left + 200, header_rect.top + 8),
        )
        self.screen.blit(
            self.font.render(mode_text, True, self.TEXT),
            (header_rect.left + 420, header_rect.top + 8),
        )
        self.screen.blit(
            self.small_font.render(help_text, True, self.MUTED),
            (header_rect.left + 14, header_rect.top + 38),
        )

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
