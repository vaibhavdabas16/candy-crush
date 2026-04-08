from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import load_ppo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DQN_PATH = ROOT / "models" / "dqn.pt"
DEFAULT_PPO_PATH = ROOT / "models" / "ppo"


def resolve_checkpoint_path(path: str | Path, suffix: str | None = None) -> Path | None:
    """Return the existing checkpoint path, accepting SB3-style paths without .zip."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if candidate.exists():
        return candidate
    if suffix is not None:
        suffixed = Path(str(candidate) + suffix)
        if suffixed.exists():
            return suffixed
    return None


def load_saved_dqn(path: str | Path = DEFAULT_DQN_PATH, device: str | None = None) -> DQNAgent:
    checkpoint_path = resolve_checkpoint_path(path)
    if checkpoint_path is None:
        raise FileNotFoundError(f"DQN model not found: {Path(path)}")
    return DQNAgent.load(checkpoint_path, device=device)


def load_saved_ppo(
    path: str | Path = DEFAULT_PPO_PATH,
    env: Any = None,
    prefer_maskable: bool = True,
):
    checkpoint_path = resolve_checkpoint_path(path, ".zip")
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"PPO model not found: {Path(path)} or {Path(str(path) + '.zip')}"
        )
    return load_ppo(checkpoint_path, env=env, prefer_maskable=prefer_maskable)


def load_saved_policy(
    name: str,
    *,
    env: Any = None,
    dqn_path: str | Path = DEFAULT_DQN_PATH,
    ppo_path: str | Path = DEFAULT_PPO_PATH,
    device: str | None = None,
    prefer_maskable: bool = True,
):
    name = name.lower()
    if name == "dqn":
        return load_saved_dqn(dqn_path, device=device)
    if name == "ppo":
        return load_saved_ppo(ppo_path, env=env, prefer_maskable=prefer_maskable)
    raise ValueError(f"Unsupported saved policy: {name}")


def saved_policy_exists(
    name: str,
    *,
    dqn_path: str | Path = DEFAULT_DQN_PATH,
    ppo_path: str | Path = DEFAULT_PPO_PATH,
) -> bool:
    name = name.lower()
    if name == "dqn":
        return resolve_checkpoint_path(dqn_path) is not None
    if name == "ppo":
        return resolve_checkpoint_path(ppo_path, ".zip") is not None
    return False
