from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """Same CNN backbone as QNetwork but outputs action logits (not Q-values)."""

    GRID: int = 8
    CANDY_TYPES: int = 6

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        self.conv = nn.Sequential(
            nn.Conv2d(self.CANDY_TYPES, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        conv_out = 64 * self.GRID * self.GRID
        self.fc = nn.Sequential(
            nn.Linear(conv_out + 1, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_dim)  →  logits: (B, action_dim)"""
        board      = obs[:, :-1].long().clamp(0, self.CANDY_TYPES - 1)  # (B, 64)
        moves_left = obs[:, -1:]                                          # (B, 1)
        one_hot    = F.one_hot(board, self.CANDY_TYPES).float()           # (B, 64, 6)
        one_hot    = one_hot.view(-1, self.GRID, self.GRID, self.CANDY_TYPES)
        one_hot    = one_hot.permute(0, 3, 1, 2)                         # (B, 6, 8, 8)
        x          = self.conv(one_hot).flatten(1)                        # (B, conv_out)
        x          = torch.cat([x, moves_left], dim=1)
        return self.fc(x)


class GRPOAgent:
    def __init__(self, obs_dim: int, action_dim: int, device: str | None = None) -> None:
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.device     = device or ("mps" if torch.backends.mps.is_available() else
                                     "cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(obs_dim, action_dim).to(self.device)

    # ── inference ─────────────────────────────────────────────────────────────
    def predict(self, obs: np.ndarray, env=None, deterministic: bool = True) -> tuple[int, None]:
        obs_t  = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(obs_t)[0]

        # Mask invalid actions
        if env is not None:
            valid = env.valid_actions()
            if valid:
                mask = torch.full((self.action_dim,), float("-inf"), device=self.device)
                mask[valid] = 0.0
                logits = logits + mask

        if deterministic:
            action = int(logits.argmax())
        else:
            probs  = F.softmax(logits, dim=-1)
            action = int(torch.multinomial(probs, 1))

        return action, None

    def action_log_prob(self, obs_t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Log-prob of actions under current policy. obs_t: (B, obs_dim), actions: (B,)"""
        logits    = self.policy(obs_t)                           # (B, action_dim)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

    # ── save / load ───────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "obs_dim":    self.obs_dim,
            "action_dim": self.action_dim,
            "policy":     self.policy.state_dict(),
        }, path)
        print(f"Saved GRPO agent → {path}")

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "GRPOAgent":
        ckpt  = torch.load(path, map_location=device or "cpu", weights_only=False)
        agent = cls(ckpt["obs_dim"], ckpt["action_dim"], device=device)
        agent.policy.load_state_dict(ckpt["policy"])
        agent.policy.eval()
        return agent
