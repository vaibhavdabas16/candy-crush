# Candy Crush RL
hi
A complete Week 1 reinforcement learning project for an 8x8 Candy Crush-style environment.

Implemented:

- Gymnasium-compatible `CandyEnv`
- 112 swap actions: 56 horizontal and 56 vertical
- Match clearing, gravity, refill, and cascades
- Rewards: `removed_candies^2 + 10` per cascade step
- Invalid swap penalty: `-5`
- DQN with replay buffer, target network, epsilon-greedy exploration, and action masking
- PPO via Stable-Baselines3, with Maskable PPO from `sb3-contrib` when available
- Random and greedy baseline policies
- Training and evaluation scripts
- TensorBoard logging for DQN and PPO
- ASCII debug rendering with `env.render(mode="human")`
- Pygame GUI for manual play and agent visualization
- Mock LLM/GRPO scaffolding for future experiments

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Run A Smoke Test

```bash
python main.py
```

## Train DQN

```bash
python train/train_dqn.py
```

Useful shorter run:

```bash
python train/train_dqn.py --episodes 50
```

The model is saved to `models/dqn.pt`, and rewards are logged to `logs/dqn_rewards.csv`.
TensorBoard logs are written under `logs/tensorboard/dqn-*`.

## Train PPO

```bash
python train/train_ppo.py
```

Useful shorter run:

```bash
python train/train_ppo.py --timesteps 5000
```

By default, the script tries Maskable PPO first. If `sb3-contrib` is unavailable, it falls back to regular PPO. You can force regular PPO with:

```bash
python train/train_ppo.py --no-maskable
```

The model is saved to `models/ppo.zip`, and rewards are logged to `logs/ppo_rewards.csv`.
TensorBoard logs are written under `logs/tensorboard/ppo-*`.

## TensorBoard

Both training scripts create a fresh TensorBoard run directory by default:

```bash
tensorboard --logdir logs/tensorboard/
```

You can choose another base directory with:

```bash
python train/train_dqn.py --log_dir logs/tensorboard
python train/train_ppo.py --log_dir logs/tensorboard
```

DQN logs episode reward, moving average reward, loss, and epsilon. PPO logs SB3 training metrics including value loss, policy gradient loss, entropy loss, and custom episode rewards.

## Debug Rendering

```python
from env.candy_env import CandyEnv

env = CandyEnv()
env.reset(seed=0)
env.render(mode="human")
```

## GUI

Manual mode:

```bash
python run_gui.py
```

Agent modes:

```bash
python run_gui.py --agent random
python run_gui.py --agent greedy
python run_gui.py --agent dqn
python run_gui.py --agent ppo
```

Controls:

- Click two adjacent candies to swap in manual mode
- `N`: random valid step in manual mode
- `R`: reset
- `Esc`: quit

The GUI uses the real `CandyEnv` for all game logic and adds simple swap, clear, and fall animations.

## Mock LLM / GRPO Stub

```bash
python train/train_grpo_stub.py
```

The stub samples valid candidate actions, simulates immediate rewards, picks the best candidate, and writes `logs/grpo_stub.csv`. It does not call any API and does not perform policy optimization.

## Evaluate

```bash
python eval/evaluate.py
```

This evaluates:

- Random policy
- Greedy policy
- DQN, if `models/dqn.pt` exists
- PPO, if `models/ppo.zip` exists

Metrics reported:

- Average score
- Score per move
- Variance across seeded episodes

## Project Structure

```text
project_root/
├── env/
│   └── candy_env.py
├── agents/
│   ├── dqn_agent.py
│   ├── ppo_agent.py
│   └── baselines.py
├── train/
│   ├── train_dqn.py
│   └── train_ppo.py
├── eval/
│   └── evaluate.py
├── utils/
│   ├── seed.py
│   └── metrics.py
├── requirements.txt
├── README.md
└── main.py
```

## Environment Details

Observation is a flat float vector of length 65:

- 64 normalized board values
- 1 normalized remaining-moves value

The board itself is stored as an 8x8 integer matrix with candies `0` through `5`.

Actions are encoded as:

- `0..55`: horizontal swaps
- `56..111`: vertical swaps

Invalid actions consume one move and return `-5`.

Action masking:

- `env.get_action_mask()` returns an `int8` binary vector of length 112
- `env.action_masks()` returns the boolean mask used by Maskable PPO
- if a board somehow has no valid move, the mask falls back to all actions so training does not crash
