from __future__ import annotations

from agents.baselines import GreedyPolicy, RandomPolicy
from env.candy_env import CandyEnv


def play_policy(policy, episodes: int = 3, seed: int = 0) -> None:
    env = CandyEnv(max_moves=20, render_mode="ansi")
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        info = {"score": 0.0}
        while not done:
            action, _ = policy.predict(obs, env=env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        print(f"episode={episode + 1} score={info['score']:.1f}")
    print(env.render())


if __name__ == "__main__":
    print("Random policy smoke test")
    play_policy(RandomPolicy(), episodes=1)
    print("\nGreedy policy smoke test")
    play_policy(GreedyPolicy(), episodes=1)
