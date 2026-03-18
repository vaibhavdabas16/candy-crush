from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        self.buffer.append((obs, action, reward, next_obs, done, next_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones, next_masks = map(np.array, zip(*batch))
        return obs, actions, rewards, next_obs, dones, next_masks

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNConfig:
    gamma: float = 0.9
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    target_update_every: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: DQNConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or DQNConfig()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.q_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = Adam(self.q_net.parameters(), lr=self.config.lr)
        self.replay = ReplayBuffer(self.config.buffer_size)
        self.steps = 0

    def epsilon(self) -> float:
        fraction = min(1.0, self.steps / float(self.config.epsilon_decay_steps))
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def select_action(
        self,
        obs: np.ndarray,
        valid_mask: np.ndarray | None = None,
        training: bool = True,
    ) -> int:
        if training and random.random() < self.epsilon():
            if valid_mask is not None and valid_mask.any():
                return int(random.choice(np.flatnonzero(valid_mask)))
            return random.randrange(self.action_dim)

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(obs_t).squeeze(0).detach().cpu().numpy()

        if valid_mask is not None and valid_mask.any():
            q_values = q_values.copy()
            q_values[~valid_mask.astype(bool)] = -np.inf
        return int(np.argmax(q_values))

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        self.replay.push(obs, action, reward, next_obs, done, next_mask)

    def update(self) -> float | None:
        if len(self.replay) < self.config.batch_size:
            return None

        obs, actions, rewards, next_obs, dones, next_masks = self.replay.sample(
            self.config.batch_size
        )
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        next_masks_t = torch.as_tensor(next_masks.astype(bool), dtype=torch.bool, device=self.device)

        q_values = self.q_net(obs_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_obs_t)
            next_q = next_q.masked_fill(~next_masks_t, -1e9)
            max_next_q = next_q.max(dim=1).values
            max_next_q = torch.where(max_next_q < -1e8, torch.zeros_like(max_next_q), max_next_q)
            target = rewards_t + self.config.gamma * (1.0 - dones_t) * max_next_q

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.config.target_update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "config": self.config.__dict__,
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "steps": self.steps,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "DQNAgent":
        try:
            checkpoint = torch.load(path, map_location=device or "cpu", weights_only=True)
        except Exception:
            try:
                checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(path, map_location=device or "cpu")
        agent = cls(
            checkpoint["obs_dim"],
            checkpoint["action_dim"],
            DQNConfig(**checkpoint["config"]),
            device=device,
        )
        agent.q_net.load_state_dict(checkpoint["q_net"])
        agent.target_net.load_state_dict(checkpoint["target_net"])
        agent.steps = checkpoint["steps"]
        return agent

    def predict(self, obs: np.ndarray, env=None, deterministic: bool = True) -> tuple[int, None]:
        mask = env.action_masks() if env is not None else None
        return self.select_action(obs, mask, training=False), None
