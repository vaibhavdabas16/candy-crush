"""
GRPO training for Candy Crush.

For each update step:
  1. Run G complete episodes from the same starting seed using the current policy.
  2. Compute total reward per episode.
  3. Group-relative advantage: A_i = (R_i - mean_R) / std_R
  4. Policy gradient loss (PPO-clip) + KL penalty vs frozen reference policy.
  5. Update policy.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import copy
import numpy as np
import torch
import torch.nn.functional as F

from env.candy_env import CandyEnv
from agents.grpo_agent import GRPOAgent

# ── Hyperparams ───────────────────────────────────────────────────────────────
G           = 8       # episodes sampled per seed
EPOCHS      = 300     # number of GRPO update steps
LR          = 3e-4
EPS_CLIP    = 0.2
BETA_KL     = 0.01
MAX_MOVES   = 20
SAVE_PATH   = Path("models/grpo_candy.pt")

device = ("mps" if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Rollout ───────────────────────────────────────────────────────────────────
def run_episode(agent: GRPOAgent, env: CandyEnv, seed: int) -> tuple[list, list, float]:
    """Run one episode. Returns (obs_list, action_list, total_reward)."""
    obs, _ = env.reset(seed=seed)
    obs_list, act_list = [], []
    total_r = 0.0
    done = False
    while not done:
        action, _ = agent.predict(obs, env=env, deterministic=False)
        obs_list.append(obs.copy())
        act_list.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_r += reward
        done = terminated or truncated
    return obs_list, act_list, total_r


# ── GRPO update ───────────────────────────────────────────────────────────────
def grpo_update(agent: GRPOAgent, ref_agent: GRPOAgent, optimizer, env: CandyEnv, seed: int):
    # Step 1 — sample G episodes (no grad)
    all_obs, all_acts, rewards = [], [], []
    for _ in range(G):
        obs_l, act_l, R = run_episode(agent, env, seed)
        all_obs.append(obs_l)
        all_acts.append(act_l)
        rewards.append(R)

    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

    # Step 2 — group-relative advantages
    mean_r = rewards_t.mean()
    std_r  = rewards_t.std().clamp(min=1e-8)
    advantages = (rewards_t - mean_r) / std_r   # (G,)

    # Step 3 — policy gradient per episode
    total_loss = torch.tensor(0.0, device=device)
    n_steps    = 0

    for g in range(G):
        if not all_obs[g]:
            continue
        obs_t  = torch.tensor(np.array(all_obs[g]),  dtype=torch.float32, device=device)
        acts_t = torch.tensor(all_acts[g], dtype=torch.long, device=device)
        adv    = advantages[g]

        new_lp  = agent.action_log_prob(obs_t, acts_t)        # (T,)
        with torch.no_grad():
            old_lp = agent.action_log_prob(obs_t, acts_t)     # (T,) — old ≈ new since we just sampled
            ref_lp = ref_agent.action_log_prob(obs_t, acts_t) # (T,)

        ratio   = torch.exp(new_lp - old_lp.detach())
        clipped = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP)
        pg_loss = -torch.min(ratio * adv, clipped * adv).mean()

        kl      = (new_lp - ref_lp.detach()).mean()
        total_loss = total_loss + pg_loss + BETA_KL * kl
        n_steps   += 1

    if n_steps > 0:
        total_loss = total_loss / n_steps
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)
        optimizer.step()

    return total_loss.item(), float(mean_r)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    env = CandyEnv(max_moves=MAX_MOVES)
    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("=" * 55)
    print("GRPO Candy Crush Training")
    print(f"obs_dim={obs_dim}  action_dim={action_dim}  G={G}")
    print(f"Epochs={EPOCHS}  LR={LR}  clip={EPS_CLIP}  kl={BETA_KL}")
    print("=" * 55)

    agent     = GRPOAgent(obs_dim, action_dim, device=device)
    ref_agent = copy.deepcopy(agent)
    ref_agent.policy.eval()
    for p in ref_agent.policy.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in agent.policy.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(agent.policy.parameters(), lr=LR)

    best_reward = -float("inf")
    rng = np.random.default_rng(42)

    for epoch in range(1, EPOCHS + 1):
        seed = int(rng.integers(0, 100_000))
        loss, mean_r = grpo_update(agent, ref_agent, optimizer, env, seed)

        if mean_r > best_reward:
            best_reward = mean_r
            agent.save(SAVE_PATH)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | loss={loss:.4f} | mean_reward={mean_r:.2f} | best={best_reward:.2f}")

    print(f"\nDone. Best reward={best_reward:.2f}. Model saved → {SAVE_PATH}")
    print("\nRun GUI with:")
    print("  python run_gui.py --agent grpo --grpo-path models/grpo_candy.pt")


if __name__ == "__main__":
    main()
