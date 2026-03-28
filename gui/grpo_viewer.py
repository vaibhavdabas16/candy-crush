"""
Post-training GRPO inference visualizer.
Shows the trained model generating token sequences one at a time,
token by token, with reward displayed per run.
"""

from __future__ import annotations
import time
import pygame


# ── Palette (matches CandyViewer) ────────────────────────────────────────────
BG     = (18,  20,  28)
PANEL  = (30,  34,  46)
BORDER = (70,  78,  98)
TEXT   = (240, 241, 245)
MUTED  = (174, 179, 190)
ACCENT = (255, 215,  96)
GREEN  = ( 75, 190, 110)
RED    = (235,  72,  78)
BLUE   = ( 63, 145, 245)
WHITE  = (255, 255, 255)
DIM    = ( 80,  88, 110)

TOKEN_COLORS = [
    (235,  72,  78),
    ( 75, 190, 110),
    ( 63, 145, 245),
    (248, 214,  76),
    (176,  95, 224),
    (246, 139,  57),
    (100, 200, 200),
    ( 220, 130, 180),
]


class GRPOInferenceViewer:
    W, H = 900, 560

    def __init__(self, policy, reward_fn, random_prompts_fn, vocab: int, n_runs: int = 20) -> None:
        pygame.init()
        self.policy           = policy
        self.reward_fn        = reward_fn
        self.random_prompts   = random_prompts_fn
        self.vocab            = vocab
        self.n_runs           = n_runs

        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("GRPO — Inference Simulation")
        self.clock  = pygame.time.Clock()
        self.font   = pygame.font.SysFont("consolas", 20)
        self.big    = pygame.font.SysFont("consolas", 28, bold=True)
        self.small  = pygame.font.SysFont("consolas", 15)

        self.history: list[dict] = []   # past runs {prompt, tokens, reward}
        self.run_idx  = 0
        self.paused   = False

    # ── public entry ──────────────────────────────────────────────────────────
    def run(self) -> None:
        running = True
        next_run_time = time.time()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_n:
                        next_run_time = 0   # force next run immediately

            now = time.time()
            if not self.paused and now >= next_run_time and self.run_idx < self.n_runs:
                self._do_one_run()
                next_run_time = now + 0.9   # pause between runs

            self._draw()
            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()

    # ── one inference pass ────────────────────────────────────────────────────
    def _do_one_run(self) -> None:
        import torch
        prompt = self.random_prompts(1)
        with torch.no_grad():
            gen, _ = self.policy.generate(prompt, max_new=8, temperature=0.8)
        reward = self.reward_fn(gen)[0].item()
        self.history.append({
            "prompt":  prompt[0].tolist(),
            "tokens":  gen[0].tolist(),
            "reward":  reward,
        })
        self.run_idx += 1

    # ── drawing ───────────────────────────────────────────────────────────────
    def _draw(self) -> None:
        self.screen.fill(BG)
        self._draw_header()
        self._draw_current()
        self._draw_history()
        self._draw_help()

    def _draw_header(self) -> None:
        title = self.big.render("GRPO  Inference  Simulation", True, ACCENT)
        self.screen.blit(title, (self.W // 2 - title.get_width() // 2, 14))

        sub = f"Run {min(self.run_idx, self.n_runs)} / {self.n_runs}"
        if self.paused:
            sub += "  [PAUSED]"
        s = self.font.render(sub, True, MUTED)
        self.screen.blit(s, (self.W // 2 - s.get_width() // 2, 46))

        # progress bar
        bar = pygame.Rect(40, 70, self.W - 80, 8)
        pygame.draw.rect(self.screen, PANEL, bar, border_radius=4)
        if self.n_runs > 0:
            w = int(bar.width * self.run_idx / self.n_runs)
            pygame.draw.rect(self.screen, BLUE, pygame.Rect(bar.left, bar.top, w, bar.height), border_radius=4)
        pygame.draw.rect(self.screen, BORDER, bar, width=1, border_radius=4)

    def _draw_current(self) -> None:
        if not self.history:
            msg = self.font.render("Waiting for first run...", True, MUTED)
            self.screen.blit(msg, (40, 110))
            return

        run = self.history[-1]
        panel = pygame.Rect(20, 90, self.W - 40, 160)
        pygame.draw.rect(self.screen, PANEL, panel, border_radius=12)
        pygame.draw.rect(self.screen, BORDER, panel, width=2, border_radius=12)

        # Prompt tokens
        self.screen.blit(self.small.render("PROMPT", True, MUTED), (36, 100))
        self._draw_tokens(run["prompt"], x0=36, y=118, target_tok=None, dim=True)

        # Arrow
        arrow = self.font.render("→", True, ACCENT)
        self.screen.blit(arrow, (36 + len(run["prompt"]) * 52 + 4, 116))

        # Generated tokens
        self.screen.blit(self.small.render("GENERATED", True, MUTED), (36, 148))
        self._draw_tokens(run["tokens"], x0=36, y=166, target_tok=5, dim=False)

        # Reward badge
        r = run["reward"]
        r_color = GREEN if r >= 6 else (ACCENT if r >= 3 else RED)
        badge_txt = self.big.render(f"Reward: {r:.0f} / 8", True, r_color)
        self.screen.blit(badge_txt, (self.W - badge_txt.get_width() - 36, 108))

        # Max reward bar
        bar_bg = pygame.Rect(self.W - 220, 148, 180, 18)
        bar_fill = pygame.Rect(self.W - 220, 148, int(180 * r / 8), 18)
        pygame.draw.rect(self.screen, DIM,     bar_bg,   border_radius=5)
        pygame.draw.rect(self.screen, r_color, bar_fill, border_radius=5)
        pygame.draw.rect(self.screen, BORDER,  bar_bg,   width=1, border_radius=5)

    def _draw_tokens(self, tokens, x0, y, target_tok, dim) -> None:
        for i, tok in enumerate(tokens):
            x = x0 + i * 52
            rect = pygame.Rect(x, y, 44, 36)
            highlight = (not dim) and (tok == target_tok)
            color = TOKEN_COLORS[tok % len(TOKEN_COLORS)]
            bg = color if highlight else (PANEL if dim else (*[c // 3 for c in color],))
            pygame.draw.rect(self.screen, bg, rect, border_radius=7)
            pygame.draw.rect(self.screen, color if not dim else BORDER, rect, width=2, border_radius=7)
            label = self.small.render(str(tok), True, WHITE if highlight else (TEXT if not dim else MUTED))
            self.screen.blit(label, label.get_rect(center=rect.center))

    def _draw_history(self) -> None:
        if len(self.history) < 2:
            return

        self.screen.blit(self.small.render("HISTORY", True, MUTED), (20, 268))

        # Scrolling reward bars for last N runs
        visible = self.history[-26:]
        bar_w   = max(8, (self.W - 80) // max(len(visible), 1) - 3)
        chart_h = 160
        base_y  = self.H - 90

        for i, run in enumerate(visible):
            r = run["reward"]
            h = int(chart_h * r / 8)
            x = 40 + i * (bar_w + 3)
            bg_rect   = pygame.Rect(x, base_y - chart_h, bar_w, chart_h)
            fill_rect = pygame.Rect(x, base_y - h,       bar_w, h)
            color = GREEN if r >= 6 else (ACCENT if r >= 3 else RED)
            pygame.draw.rect(self.screen, DIM,   bg_rect,   border_radius=3)
            pygame.draw.rect(self.screen, color, fill_rect, border_radius=3)

            # Mark latest
            if i == len(visible) - 1:
                pygame.draw.rect(self.screen, WHITE, fill_rect, width=1, border_radius=3)

        # Axis labels
        self.screen.blit(self.small.render("8", True, MUTED), (20, base_y - chart_h))
        self.screen.blit(self.small.render("0", True, MUTED), (20, base_y - 14))

        # Average line
        avg = sum(r["reward"] for r in visible) / len(visible)
        avg_y = base_y - int(chart_h * avg / 8)
        pygame.draw.line(self.screen, ACCENT, (40, avg_y), (40 + len(visible) * (bar_w + 3), avg_y), 1)
        self.screen.blit(self.small.render(f"avg={avg:.1f}", True, ACCENT), (self.W - 90, avg_y - 14))

    def _draw_help(self) -> None:
        help_txt = "SPACE pause/resume   N next run   ESC quit"
        s = self.small.render(help_txt, True, DIM)
        self.screen.blit(s, (self.W // 2 - s.get_width() // 2, self.H - 20))
