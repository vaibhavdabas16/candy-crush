# Candy Crush RL

A complete reinforcement learning project for an 8×8 Candy Crush-style environment, with classical RL baselines (random, greedy, DQN, PPO) and an LLM-based policy (Qwen 9B fine-tuned with GRPO, served as a Q4_K_M GGUF for CPU/Mac inference).

The trained models are already in this repository (`models/dqn.pt`, `models/ppo.zip`); the GRPO GGUF is downloaded on demand from Hugging Face. **No training is required to run the demos.**

---

## Quick Start (one command, fresh Ubuntu 22.04)

```bash
bash run.sh
```

That single script runs the **entire pipeline** end-to-end in a fresh `ubuntu:22.04` container, with no manual install, no manual configuration, and only relative paths.

### What `run.sh` does, in order

| # | Stage | Wall-clock |
| --- | --- | --- |
| 1 | apt-install `python3-venv`, `build-essential`, `git`, `ca-certificates` | seconds |
| 2 | Create `.venv`, install baseline + GRPO-inference deps (gymnasium, torch, sb3, sb3-contrib, llama-cpp-python, huggingface_hub) | minutes (one-time) |
| 3 | Train **DQN** from scratch | ~5 min |
| 4 | Train **PPO** from scratch | ~5 min |
| 5 | **Run the trained DQN/PPO + random + greedy** on the same seeds, print every step's board + action, finish with a comparison table | seconds |
| 6 | Download the merged Qwen GRPO **Q4_K_M GGUF** (~5.3 GB, one-time) and play it on the same seeds, with greedy as a sanity column | minutes per seed |
| 7 | **(END) GRPO LoRA training** — install the heavy training deps and run for up to **1 hour** | up to 1 h |

Stages 3 and 4 produce `models/dqn.pt` and `models/ppo.zip`; stage 5 immediately re-uses those freshly trained checkpoints. Stage 6 uses the canonical trained GRPO model from Hugging Face (`arnavm7/candy-crush-qwen35-grpo-lora`, the merged Q4_K_M GGUF) — that 1-hour training in stage 7 will not converge on CPU and is part of the pipeline structure only; the real model is the one downloaded in stage 6.

Every stage tee's its output to a file under `./logs/` so the evaluator can read the full transcript afterwards.

### What each stage prints

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

CLI flags are forwarded straight to the demo scripts. Common overrides:

```bash
bash run.sh --seeds 10 11 12 --max-moves 30      # different seeds / longer episodes
bash run.sh --seeds 0 --max-moves 5              # quick smoke test
bash run.sh --no-greedy                          # skip the greedy comparison column

# Cap the length of the two short training stages with env vars (defaults shown):
DQN_EPISODES=200 PPO_TIMESTEPS=30000 bash run.sh

# Use Metal/CUDA for GRPO inference (default is CPU so a clean docker works):
bash run.sh --gguf-n-gpu-layers -1
```

---

## Results

Fixed-board GRPO eval protocol: 10 special-candy boards with `seed ∈ {20000..20009}` and `special_seed = seed + 50000`, each played as a full 20-move rollout. The GRPO policy runs without the greedy-fallback safety net — every parse failure or illegal swap counts as a real invalid action and gets the env's `-5` penalty.

```
Policy     | avg ± std         | min  | max  | invalid_rate
-----------+-------------------+------+------+----------------------------------
random     | 1072.90 ± 218.06  |  697 | 1383 | 0.000
ppo        |  978.50 ± 273.60  |  583 | 1460 | 0.000
dqn        | 1092.80 ± 233.33  |  668 | 1596 | 0.000
grpo_gguf  | 1827.80 ± 569.18  | 1193 | 3005 | 0.005   parse_invalid=0.000  model_invalid=0.005
greedy     | 2314.60 ± 548.53  | 1511 | 3100 | 0.000
```

GRPO emitted **0 parse failures and 1 illegal swap across 200 model decisions** (199/200 = 99.5% legal-action rate from the LLM alone, no fallback). It outperformed random / DQN / PPO on 7 of 10 boards and trailed greedy on most boards but won outright on board 20008 (3005 vs 2940).

Per-board total reward:

| seed  | random | ppo  | dqn  | grpo_gguf | greedy |
|-------|-------:|-----:|-----:|----------:|-------:|
| 20000 |  1383  |  781 |  979 |    1258   |  1511  |
| 20001 |  1266  |  808 | 1210 |    1193   |  2486  |
| 20002 |   697  |  835 | 1142 |    2371   |  2387  |
| 20003 |   854  |  583 | 1156 |    1872   |  1916  |
| 20004 |  1026  | 1460 |  906 |    2209   |  1813  |
| 20005 |   958  | 1378 |  956 |    1810   |  2206  |
| 20006 |  1371  | 1014 | 1596 |    2034   |  3100  |
| 20007 |   983  |  715 |  668 |    1224   |  1728  |
| 20008 |   955  | 1038 | 1069 |    3005   |  2940  |
| 20009 |  1236  | 1173 | 1246 |    1302   |  3059  |

Reproduce with:
```bash
python eval/evaluate.py --policies random greedy dqn ppo grpo_gguf \
  --gguf-n-gpu-layers -1 --json-out logs/eval.json
```

(`--gguf-n-gpu-layers -1` offloads to Metal/CUDA. Use `0` for CPU only — adds ~30 s/move on a CPU box.)

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
│   └── llm_grpo_gguf_agent.py # llama.cpp-backed Q4_K_M GGUF path (used in stage 6)
├── train/
│   ├── train_dqn.py           # Stage 3: trained from scratch in run.sh
│   ├── train_ppo.py           # Stage 4: trained from scratch in run.sh
│   └── train_llm_grpo_candy.py # Stage 7: GRPO LoRA training
├── eval/evaluate.py           # Fixed-board GRPO eval protocol (10 boards, full rollouts)
├── scripts/
│   ├── play_baselines_demo.py # Multi-agent terminal demo (random/greedy/DQN/PPO)
│   └── play_grpo_demo.py      # GRPO terminal demo + greedy comparison
├── gui/viewer.py              # Pygame visual viewer
├── run_gui.py                 # GUI / terminal player (--no-gui mode)
├── models/                    # Pre-trained checkpoints already shipped in the repo
│   ├── dqn.pt
│   └── ppo.zip
├── requirements.txt              # Original full-stack reqs
├── requirements-baselines.txt    # Stage 2 deps  (baselines + GRPO inference)
├── requirements-train-grpo.txt   # Stage 7 deps  (transformers / peft / trl / datasets / accelerate)
└── run.sh                        # End-to-end pipeline (stages 1-7)
```

## The GRPO model

The Qwen GRPO model lives at:

> https://huggingface.co/arnavm7/candy-crush-qwen35-grpo-lora

Stage 6 of `run.sh` downloads the merged Q4_K_M GGUF (`gguf/candy-crush-qwen35-grpo-Q4_K_M.gguf`, ~5.3 GB) into `models/llm_grpo_candy/qwen35_9b/gguf/` on first run. The download is skipped on subsequent runs. The same Hugging Face repo also hosts the original LoRA adapter, which the older Transformers/PEFT path uses; the GGUF is the merged base + adapter for fast non-CUDA inference.

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
