"""Boosted DQN training with Greedy demonstrations + Behavioral Cloning (BC) loss.

Vanilla DQN replay of greedy-picked transitions only updates Q(s, a_greedy) — so
the argmax over all valid actions at eval time can end up on a random
non-demonstrated action. We fix this by adding a supervised BC term that pushes
Q(s, a_greedy) above the Q of all other valid actions for every demo transition.
This produces an agent that matches (and eventually surpasses) the one-step
Greedy baseline, because online fine-tuning then refines multi-step combos.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from agents.baselines import GreedyPolicy
from agents.dqn_agent import DQNAgent, DQNConfig
from env.candy_env import CandyEnv
from utils.seed import set_global_seed
from utils.tensorboard import make_run_dir


def bc_pretrain(
    agent: DQNAgent,
    demos: list[tuple[np.ndarray, int, np.ndarray]],
    epochs: int,
    batch_size: int,
    bc_margin: float,
    writer,
) -> None:
    """Supervised pre-training: push Q(s, greedy_action) above Q(s, others) by bc_margin."""
    if not demos:
        return
    obs_all = np.stack([d[0] for d in demos]).astype(np.float32)
    act_all = np.array([d[1] for d in demos], dtype=np.int64)
    mask_all = np.stack([d[2].astype(bool) for d in demos])

    opt = Adam(agent.q_net.parameters(), lr=1e-3)
    device = agent.device
    n = len(demos)
    for epoch in range(epochs):
        idx = np.random.permutation(n)
        epoch_loss = 0.0
        steps = 0
        for start in range(0, n, batch_size):
            batch = idx[start:start + batch_size]
            obs_t = torch.as_tensor(obs_all[batch], device=device)
            act_t = torch.as_tensor(act_all[batch], device=device)
            mask_t = torch.as_tensor(mask_all[batch], device=device)

            q = agent.q_net(obs_t)
            demo_q = q.gather(1, act_t.unsqueeze(1)).squeeze(1)

            # Margin loss: every valid non-demo action should be <= demo_q - margin
            q_masked = q.masked_fill(~mask_t, -1e9)
            q_masked.scatter_(1, act_t.unsqueeze(1), -1e9)  # exclude demo action
            max_other = q_masked.max(dim=1).values
            has_alt = (mask_t.sum(dim=1) > 1)
            margin_loss = torch.clamp(
                max_other + bc_margin - demo_q, min=0.0
            )
            margin_loss = (margin_loss * has_alt.float()).mean()

            opt.zero_grad()
            margin_loss.backward()
            nn.utils.clip_grad_norm_(agent.q_net.parameters(), 5.0)
            opt.step()
            epoch_loss += float(margin_loss.item()); steps += 1
        avg = epoch_loss / max(steps, 1)
        writer.add_scalar("bc_pretrain/margin_loss", avg, epoch)
        if epoch % 5 == 0:
            print(f"bc epoch={epoch} margin_loss={avg:.4f}")
    agent.target_net.load_state_dict(agent.q_net.state_dict())


def collect_greedy_demonstrations(
    agent: DQNAgent, n_episodes: int, max_moves: int, seed: int
) -> list[tuple[np.ndarray, int, np.ndarray]]:
    env = CandyEnv(max_moves=max_moves)
    policy = GreedyPolicy()
    demos: list[tuple[np.ndarray, int, np.ndarray]] = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + 10_000 + ep)
        done = False
        while not done:
            valid_mask = info["valid_action_mask"].copy()
            action, _ = policy.predict(obs, env=env)
            next_obs, reward, term, trunc, next_info = env.step(int(action))
            done = term or trunc
            agent.store(obs, int(action), float(reward), next_obs, done, next_info["valid_action_mask"])
            demos.append((obs.copy(), int(action), valid_mask))
            obs = next_obs
            info = next_info
    return demos


def train(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    env = CandyEnv(max_moves=args.max_moves)
    from torch.utils.tensorboard import SummaryWriter

    tb_run_dir = make_run_dir(ROOT / args.log_dir, "dqn_boosted")
    writer = SummaryWriter(log_dir=str(tb_run_dir))

    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=DQNConfig(
            gamma=args.gamma,
            lr=args.lr,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            epsilon_start=0.3,
            epsilon_end=0.03,
            epsilon_decay_steps=args.epsilon_decay_steps,
            target_update_every=args.target_update_every,
        ),
    )

    print(f"Pre-filling replay buffer with {args.demo_episodes} Greedy demonstrations...")
    demos = collect_greedy_demonstrations(agent, args.demo_episodes, args.max_moves, args.seed)
    print(f"Replay buffer size after demos: {len(agent.replay)}  |  BC pairs: {len(demos)}")

    print("Behavioral Cloning pretraining (margin-based)...")
    bc_pretrain(agent, demos, epochs=args.bc_epochs, batch_size=256, bc_margin=args.bc_margin, writer=writer)

    print("Q-learning pretraining on demo transitions...")
    for step in range(args.pretrain_updates):
        loss = agent.update()
        if step % 500 == 0 and loss is not None:
            print(f"pretrain step={step} loss={loss:.4f}")
            writer.add_scalar("pretrain/loss", loss, step)

    recent_rewards: list[float] = []
    rewards_log: list[dict] = []
    global_step = 0
    for episode in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + episode)
        done = False
        episode_reward = 0.0
        losses: list[float] = []
        while not done:
            action = agent.select_action(obs, info["valid_action_mask"], training=True)
            next_obs, reward, term, trunc, next_info = env.step(action)
            done = term or trunc
            agent.store(obs, action, reward, next_obs, done, next_info["valid_action_mask"])
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            obs = next_obs
            info = next_info
            episode_reward += reward
            global_step += 1

        recent_rewards.append(episode_reward)
        if len(recent_rewards) > args.ma_window:
            recent_rewards.pop(0)
        mean_loss = sum(losses) / len(losses) if losses else 0.0
        ma = sum(recent_rewards) / len(recent_rewards)
        row = {"episode": episode, "reward": episode_reward, "ma": ma, "loss": mean_loss, "eps": agent.epsilon()}
        rewards_log.append(row)
        writer.add_scalar("train/episode_reward", episode_reward, episode)
        writer.add_scalar("train/moving_average_reward", ma, episode)
        writer.add_scalar("train/loss", mean_loss, episode)
        if episode % args.log_every == 0 or episode == 1:
            print(f"episode={episode} reward={episode_reward:.1f} ma={ma:.1f} loss={mean_loss:.4f} eps={agent.epsilon():.3f}")

    writer.flush(); writer.close()
    model_path = ROOT / args.model_path
    agent.save(model_path)
    log_path = ROOT / args.log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "reward", "ma", "loss", "eps"])
        w.writeheader(); w.writerows(rewards_log)
    print(f"Saved DQN model to {model_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--demo-episodes", type=int, default=400)
    p.add_argument("--pretrain-updates", type=int, default=4000)
    p.add_argument("--max-moves", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gamma", type=float, default=0.92)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=80_000)
    p.add_argument("--epsilon-decay-steps", type=int, default=12_000)
    p.add_argument("--target-update-every", type=int, default=500)
    p.add_argument("--ma-window", type=int, default=20)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--model-path", type=str, default="models/dqn.pt")
    p.add_argument("--log-path", type=str, default="logs/dqn_boosted_rewards.csv")
    p.add_argument("--log_dir", type=str, default="logs/tensorboard")
    p.add_argument("--bc-epochs", type=int, default=40)
    p.add_argument("--bc-margin", type=float, default=5.0)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
