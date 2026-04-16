from __future__ import annotations

from pathlib import Path


def make_ppo(env, gamma: float = 0.9, use_maskable: bool = True, seed: int = 0):
    if use_maskable:
        try:
            from sb3_contrib import MaskablePPO

            return MaskablePPO(
                "MlpPolicy",
                env,
                gamma=gamma,
                verbose=1,
                seed=seed,
                n_steps=512,
                batch_size=64,
            )
        except ImportError:
            pass

    from stable_baselines3 import PPO

    return PPO(
        "MlpPolicy",
        env,
        gamma=gamma,
        verbose=1,
        seed=seed,
        n_steps=512,
        batch_size=64,
    )


def load_ppo(path: str | Path, env=None, prefer_maskable: bool = True):
    if prefer_maskable:
        try:
            from sb3_contrib import MaskablePPO

            return MaskablePPO.load(path, env=env)
        except Exception:
            pass

    from stable_baselines3 import PPO

    return PPO.load(path, env=env)
