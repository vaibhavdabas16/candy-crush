# Candy Crush RL

A complete reinforcement learning project for an 8×8 Candy Crush-style environment, with classical RL baselines (random, greedy, DQN, PPO) and an LLM-based policy (Qwen 9B fine-tuned with GRPO, served as a Q4_K_M GGUF for CPU/Mac inference).

The trained models are already in this repository (`models/dqn.pt`, `models/ppo.zip`); the GRPO GGUF is downloaded on demand from Hugging Face. **No training is required to run the demos.**

---

## Quick Start (one command, fresh Ubuntu 22.04)

Two end-to-end bash scripts. Each one creates its own venv, installs everything, and runs a full demo with a comparison table.

```bash
# 1) Baselines: random, greedy, DQN, PPO. Fast (~30 s on a clean machine).
bash run.sh

# 2) GRPO: heavy. Downloads a 5.3 GB GGUF, then runs the 9B model on CPU.
#    Plan for ~3-5 minutes per seed of pure inference.
bash run_grpo.sh
```

Both scripts use **only relative paths**, install only what they need (no leftover transformers/peft/datasets/pygame for the baselines run), and write the full terminal transcript under `./logs/`.

### Why two scripts?

The two paths have very different cost profiles, dependency sets, and runtime characteristics — keeping them apart means you can sanity-check the baselines in seconds without paying the GRPO cost, and the GRPO run is isolated in its own venv.

| | `run.sh` (baselines) | `run_grpo.sh` (GRPO) |
| --- | --- | --- |
| Runtime | seconds per agent per seed | ~30-60 s per move (LLM inference) |
| Disk | ~1.5 GB venv | ~5.3 GB GGUF download + ~500 MB venv |
| Dependencies | `gymnasium`, `numpy`, `torch`, `stable-baselines3`, `sb3-contrib` | `gymnasium`, `numpy`, `huggingface_hub`, `llama-cpp-python` |
| Models needed | `models/dqn.pt`, `models/ppo.zip` (already in repo) | `arnavm7/candy-crush-qwen35-grpo-lora` GGUF (auto-downloaded) |
| venv | `.venv` | `.venv-grpo` |
| Network | none | Hugging Face download on first run |

`random` and `greedy` need no checkpoints at all; they're included in `run.sh` because they share the same dependency set as DQN/PPO.

### What each script prints

For every step, the ANSI board state followed by the action the agent took, the reward, and the running score — exactly what the Pygame GUI shows visually, but in the terminal so it works in a docker container with no display:

```
[step 1]
Score: 0.0 | Moves left: 20 | Last action: None
5. 3. 5. 1. 1. 0. 3. 4.
1. 4. 5. 5. 3. 3. 5. 4.
...
action=40 swap (5, 5) <-> (5, 6)  reward=+65.00 score=65.00  moves_left=19
```

For GRPO the agent also prints the raw model output and the parsed action:

```
[grpo-gguf step=1 14.21s] raw='swap (5, 5) (5, 6)\nReason: This swap creates a match of three' parsed_action=40 valid=True chosen=40 ((5, 5), (5, 6)) [ok]
```

Each script ends with a comparison table (per-seed reward + average).

### Override flags

```bash
bash run.sh --seeds 10 11 12 --max-moves 30
bash run_grpo.sh --seeds 0 1 --max-moves 20 --gguf-n-gpu-layers -1   # use GPU
bash run_grpo.sh --no-greedy                                          # GRPO only
```

`--gguf-n-gpu-layers -1` offloads all layers to Metal/CUDA if available; the default is `0` (CPU only) so it works in a fresh docker container with no GPU.

---

## Repository layout

```
candy-crush-repo/
├── env/candy_env.py           # Gymnasium-compatible 8x8 environment (112 swap actions)
├── agents/
│   ├── baselines.py           # RandomPolicy, GreedyPolicy
│   ├── dqn_agent.py           # DQN with replay buffer, target net, action masking
│   ├── ppo_agent.py           # PPO via stable-baselines3 (Maskable PPO if available)
│   ├── grpo_agent.py          # GRPO policy
│   ├── llm_grpo_agent.py      # Transformers/PEFT path for the Qwen LoRA adapter
│   └── llm_grpo_gguf_agent.py # llama.cpp-backed Q4_K_M GGUF path (used by run_grpo.sh)
├── train/                     # Training scripts (NOT used by run.sh / run_grpo.sh)
├── eval/evaluate.py           # Fixed-board GRPO eval protocol (10 boards, full rollouts)
├── scripts/
│   ├── play_baselines_demo.py # Multi-agent terminal demo (random/greedy/DQN/PPO)
│   └── play_grpo_demo.py      # GRPO terminal demo + greedy comparison
├── gui/viewer.py              # Pygame visual viewer
├── run_gui.py                 # GUI / terminal player (--no-gui mode)
├── models/                    # Pre-trained checkpoints already shipped in the repo
│   ├── dqn.pt
│   └── ppo.zip
├── requirements.txt           # Full dependency set (training + everything)
├── requirements-baselines.txt # Minimal deps for run.sh
├── requirements-grpo.txt      # Deps for run_grpo.sh
├── run.sh                     # End-to-end baseline demo
└── run_grpo.sh                # End-to-end GRPO demo
```

## The GRPO model

The Qwen GRPO model lives at:

> https://huggingface.co/arnavm7/candy-crush-qwen35-grpo-lora

`run_grpo.sh` downloads the merged Q4_K_M GGUF (`gguf/candy-crush-qwen35-grpo-Q4_K_M.gguf`, ~5.3 GB) into `models/llm_grpo_candy/qwen35_9b/gguf/` on first run. The download is skipped on subsequent runs. The same Hugging Face repo also hosts the original LoRA adapter (`arnavm7/candy-crush-qwen35-grpo-lora`), which the older Transformers/PEFT path uses; the GGUF is the merged base + adapter for fast non-CUDA inference.

---

## Pre-trained models in the repo

| File | What it is |
| --- | --- |
| `models/dqn.pt` | Trained DQN policy (replay buffer + target net + action masking) |
| `models/ppo.zip` | Trained PPO policy (stable-baselines3, Maskable PPO when available) |

These are loaded directly by `agents/saved_models.py` in the demo. **No training is required.** If you want to retrain from scratch see the `## Train DQN` and `## Train PPO` sections below — these are not part of the bash-script demo.

---

## Manual / advanced usage

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with:

```bash
.venv\Scripts\activate
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

## Reuse saved DQN/PPO checkpoints

The reusable checkpoint loaders live in `agents/saved_models.py`:

```python
from agents.saved_models import load_saved_policy
from env.candy_env import CandyEnv

env = CandyEnv(max_moves=20)
dqn = load_saved_policy("dqn")
ppo = load_saved_policy("ppo", env=env)

obs, info = env.reset(seed=0)
action, _ = ppo.predict(obs, deterministic=True, action_masks=env.action_masks())
```

By default, DQN loads from `models/dqn.pt` and PPO loads from `models/ppo.zip`.

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
python run_gui.py --agent llm_grpo
```

## Run The Qwen GRPO Model In The GUI

The trained LoRA adapter is hosted on Hugging Face:

```text
arnavm7/candy-crush-qwen35-grpo-lora
```

Fast path, using the adapter directly from Hugging Face:

```bash
python run_gui.py \
  --agent llm_grpo \
  --llm-grpo-path arnavm7/candy-crush-qwen35-grpo-lora \
  --llm-model-name Qwen/Qwen3.5-9B \
  --agent-delay 0.8
```

Local-cache path, useful for repeat runs and offline testing:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="arnavm7/candy-crush-qwen35-grpo-lora",
    local_dir="models/llm_grpo_candy/qwen35_9b/final_plus30",
    ignore_patterns=["standalone/*"],
)
PY

python run_gui.py \
  --agent llm_grpo \
  --llm-grpo-path models/llm_grpo_candy/qwen35_9b/final_plus30 \
  --llm-model-name Qwen/Qwen3.5-9B \
  --agent-delay 0.8
```

On a headless server, this verifies startup without opening a visible window:

```bash
SDL_VIDEODRIVER=dummy timeout 30 python run_gui.py \
  --agent llm_grpo \
  --llm-grpo-path arnavm7/candy-crush-qwen35-grpo-lora
```

The GUI serializes the current board to text, asks Qwen plus the LoRA adapter for a command like `swap (3,5) (3,6)`, validates the parsed swap against `CandyEnv`, and falls back to the best immediate legal swap if the model output is not parseable.

MacBook and CPU-only inference instructions are documented in [docs/mac_cpu_inference.md](docs/mac_cpu_inference.md). The most efficient non-CUDA path is the merged `Q4_K_M` GGUF documented in [docs/gguf_quantization.md](docs/gguf_quantization.md). The Transformers GUI path still uses the LoRA adapter directly; for that path use `--llm-no-4bit`, `--llm-device mps --llm-dtype float16` on Apple Silicon, and `--llm-device cpu --llm-dtype float32` on CPU-only machines.

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

## Train Qwen GRPO

The Qwen trainer runs GRPO candidates and only promotes a candidate when its fixed eval beats the incumbent. It discards failed or non-improving checkpoints and uses `beta_kl=0`, so no reference model is loaded.

```bash
python train/train_llm_grpo_candy.py \
  --model-name Qwen/Qwen3.5-9B \
  --run-dir models/llm_grpo_candy/qwen35_9b \
  --iterations 15 \
  --lora-rank 64 \
  --num-generations 8 \
  --rollout-depth 1 \
  --experiment-timeout-sec 600
```

The production adapter currently used by the GUI was trained from this flow and uploaded to Hugging Face as `arnavm7/candy-crush-qwen35-grpo-lora`.

Evaluation results and strategy notes are documented in [docs/qwen_grpo_evaluation.md](docs/qwen_grpo_evaluation.md), including the fixed eval against greedy, random, PPO, and DQN, plus expected behavior when adding a new candy type.

## Evaluate

```bash
python eval/evaluate.py
```

This evaluates:

- Random policy
- Greedy policy
- DQN, if `models/dqn.pt` exists
- PPO, if `models/ppo.zip` exists
- Qwen GRPO LoRA, when evaluated through the LLM trainer/eval path

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
