# Candy Crush RL

Reinforcement-learning agents for an 8×8 Candy Crush–style environment, with classical RL baselines (Random, Greedy, **DQN**, **PPO**) and a 9-billion-parameter **LLM policy** fine-tuned with GRPO and served as a 5.3 GB Q4_K_M GGUF for fast CPU/Metal inference.

> **One command, fresh `ubuntu:22.04`, full pipeline:** `bash run.sh`

---

## Project Demo Video

[![Project Demo Video](https://img.youtube.com/vi/TowM1f1c2gc/maxresdefault.jpg)](https://youtu.be/TowM1f1c2gc)

[Watch the project demo video](https://youtu.be/TowM1f1c2gc)

---

## Table of contents

- [Quick start](#quick-start)
- [Pipeline at a glance](#pipeline-at-a-glance)
- [Results](#results)
- [How the LLM policy works](#how-the-llm-policy-works)
- [Repository layout](#repository-layout)
- [Pre-trained artifacts](#pre-trained-artifacts)
- [Configuration](#configuration)
- [Manual usage](#manual-usage)
- [Troubleshooting](#troubleshooting)

---

## Quick start

```bash
git clone https://github.com/vaibhavdabas16/candy-crush.git
cd candy-crush
bash run.sh
```

The script is **fully self-contained**:

- Installs system packages with `apt` if missing (`python3-venv`, `build-essential`, `git`, `ca-certificates`).
- Creates a `.venv`, installs only the deps it needs, in two waves.
- Trains DQN and PPO from scratch.
- Plays every agent — Random, Greedy, **freshly trained DQN**, **freshly trained PPO** — on the same seeds and prints a comparison table.
- Downloads the merged Qwen GRPO GGUF (5.3 GB) and plays it on the same seeds, with Greedy as a sanity column.
- (At the end) attempts 1 hour of GRPO LoRA training on a small Qwen base model — the structural finale; the canonical GRPO model is the GGUF used in stage 6.

All paths are relative. All artifacts land under `./logs` and `./models`. Verified end-to-end in a fresh `ubuntu:22.04` Docker container.

---

## Pipeline at a glance

| # | Stage | Output | Wall-clock |
| --- | --- | --- | --- |
| 1 | System setup (`apt-get install python3-venv build-essential git`) | — | seconds |
| 2 | `.venv` + baseline + GRPO-inference deps | `.venv/` | minutes (one-time) |
| 3 | **Train DQN** | `models/dqn.pt`, `logs/train_dqn.log` | ~5 min |
| 4 | **Train PPO** | `models/ppo.zip`, `logs/train_ppo.log` | ~5 min |
| 5 | Demo Random / Greedy / DQN / PPO on shared seeds, print comparison | `logs/run_baselines.log` | seconds |
| 6 | Download Q4_K_M GGUF, play GRPO + Greedy on shared seeds | `models/llm_grpo_candy/qwen35_9b/gguf/…`, `logs/run_grpo_eval.log` | minutes per seed |
| 7 | **(END) GRPO LoRA training**, capped at 1 h | `models/llm_grpo_candy/run_local/`, `logs/train_grpo.log` | up to 1 h |

Stage 7 is gated to the very end on purpose: a CPU-only `ubuntu:22.04` cannot meaningfully train a 9B-parameter LLM in 1 h, so the pipeline first exercises every other stage to completion, then runs the GRPO training pipeline as a structural finale. The GRPO model evaluated in stage 6 is the **already-trained** model from Hugging Face, [`arnavm7/candy-crush-qwen35-grpo-lora`](https://huggingface.co/arnavm7/candy-crush-qwen35-grpo-lora).

---

## Results

Fixed-board GRPO eval protocol — 10 special-candy boards with `seed ∈ {20000..20009}` and `special_seed = seed + 50000`, each played as a full 20-move rollout. The GRPO policy runs **without a greedy-fallback safety net**: every parse failure or illegal swap counts as a real invalid action and gets the env's `−5` penalty.

```
Policy     | avg ± std         | min  | max  | invalid_rate
-----------+-------------------+------+------+-------------------------------
random     | 1072.90 ± 218.06  |  697 | 1383 | 0.000
ppo        |  978.50 ± 273.60  |  583 | 1460 | 0.000
dqn        | 1092.80 ± 233.33  |  668 | 1596 | 0.000
grpo_gguf  | 1827.80 ± 569.18  | 1193 | 3005 | 0.005   (parse=0.000, model=0.005)
greedy     | 2314.60 ± 548.53  | 1511 | 3100 | 0.000
```

GRPO emitted **0 parse failures and 1 illegal swap across 200 model decisions** (199/200 = 99.5 % legal-action rate from the LLM alone, no fallback). It outperformed Random / DQN / PPO on 7 of 10 boards and trailed Greedy on most boards but **won outright on board 20008** (3005 vs 2940).

<details>
<summary>Per-board total reward (click to expand)</summary>

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
| 20008 |   955  | 1038 | 1069 |  **3005** |  2940  |
| 20009 |  1236  | 1173 | 1246 |    1302   |  3059  |

</details>

Reproduce:

```bash
python eval/evaluate.py --policies random greedy dqn ppo grpo_gguf \
  --gguf-n-gpu-layers -1 --json-out logs/eval.json
```

`--gguf-n-gpu-layers -1` offloads to Metal/CUDA. Use `0` for CPU only — adds ~30 s/move on a CPU box.

---

## How the LLM policy works

```
                 ┌──────────────────────────────────────────────────┐
                 │  CandyEnv state  (board, specials, moves left)    │
                 └────────────────────────┬─────────────────────────┘
                                          │ utils.state_to_text
                                          ▼
                 ┌──────────────────────────────────────────────────┐
                 │  Text prompt:                                     │
                 │   board grid, legal actions, special-candy rules, │
                 │   "First line only the swap command:"             │
                 └────────────────────────┬─────────────────────────┘
                                          │  llama-cpp-python
                                          ▼
                 ┌──────────────────────────────────────────────────┐
                 │  Qwen 9B + GRPO LoRA (merged Q4_K_M GGUF, 5.3 GB) │
                 └────────────────────────┬─────────────────────────┘
                                          │  raw text
                                          ▼
                 ┌──────────────────────────────────────────────────┐
                 │  Parser:  swap (r,c) (r,c)  →  CandyEnv action    │
                 │  + reasoning trace (kept for the log only)        │
                 └────────────────────────┬─────────────────────────┘
                                          ▼
                 ┌──────────────────────────────────────────────────┐
                 │  env.step(action)  →  reward, new state           │
                 └──────────────────────────────────────────────────┘
```

**Inference details:**

- Merged Q4_K_M GGUF: no separate base model + LoRA adapter at runtime.
- Deterministic decoding (`temperature=0`, `max_new_tokens=24`).
- `no_fallback=True` in eval mode: parse failures and illegal swaps are logged as such and pay the env penalty, never silently rescued.
- Per-step latency: ~2-5 s on Apple Silicon Metal, ~30-60 s on a CPU-only Linux container.

The same agent class powers the GUI (`run_gui.py --agent llm_grpo_gguf`) and the terminal demo (`scripts/play_grpo_demo.py`).

---

## Repository layout

```
candy-crush/
├── env/candy_env.py               # Gymnasium-compatible 8×8 env, 112 swap actions
├── agents/
│   ├── baselines.py               # RandomPolicy, GreedyPolicy
│   ├── dqn_agent.py               # DQN: replay buffer, target net, action masking
│   ├── ppo_agent.py               # PPO via stable-baselines3 (Maskable PPO if available)
│   ├── grpo_agent.py              # Tabular GRPO baseline
│   ├── llm_grpo_agent.py          # Transformers/PEFT LoRA-adapter path
│   └── llm_grpo_gguf_agent.py     # llama.cpp Q4_K_M GGUF path  ← used in run.sh
├── train/
│   ├── train_dqn.py               # Stage 3 of run.sh
│   ├── train_ppo.py               # Stage 4 of run.sh
│   └── train_llm_grpo_candy.py    # Stage 7 of run.sh
├── eval/evaluate.py               # Fixed-board GRPO eval protocol (the table above)
├── scripts/
│   ├── play_baselines_demo.py     # Stage 5: terminal demo + comparison
│   └── play_grpo_demo.py          # Stage 6: GRPO terminal demo + comparison
├── gui/viewer.py                  # Pygame visual viewer
├── run_gui.py                     # GUI entry, also `--no-gui` terminal mode
├── models/
│   ├── dqn.pt                     # Pre-trained DQN, overwritten by stage 3
│   └── ppo.zip                    # Pre-trained PPO, overwritten by stage 4
├── docs/
│   └── gguf_quantization.md       # GGUF artifact details + how to re-create it
├── requirements.txt               # Original full-stack reqs (training etc.)
├── requirements-baselines.txt     # Stage 2: baselines + GRPO inference
├── requirements-train-grpo.txt    # Stage 7: heavy GRPO training deps
└── run.sh                         # End-to-end pipeline (the only script you need)
```

---

## Pre-trained artifacts

| Artifact | Where it lives | What it is |
| --- | --- | --- |
| `models/dqn.pt` | shipped in repo | Trained DQN policy (replay + target net + action masking). Overwritten in stage 3. |
| `models/ppo.zip` | shipped in repo | Trained PPO policy (stable-baselines3, Maskable PPO when available). Overwritten in stage 4. |
| Q4_K_M GGUF | [`arnavm7/candy-crush-qwen35-grpo-lora`](https://huggingface.co/arnavm7/candy-crush-qwen35-grpo-lora) on Hugging Face | Merged Qwen 3.5-9B + GRPO LoRA, quantized for fast non-CUDA inference. Auto-downloaded in stage 6 to `models/llm_grpo_candy/qwen35_9b/gguf/`. |

See [`docs/gguf_quantization.md`](docs/gguf_quantization.md) for GGUF size, SHA256, and how the artifact was produced.

---

## Configuration

All flags after `bash run.sh` are forwarded to the demo scripts:

```bash
bash run.sh --seeds 10 11 12 --max-moves 30
bash run.sh --seeds 0 --max-moves 5         # quick smoke test
bash run.sh --no-greedy                     # skip the greedy column in stage 6
bash run.sh --gguf-n-gpu-layers -1          # offload GGUF to Metal/CUDA
```

Cap the two short training stages with environment variables:

```bash
DQN_EPISODES=200 PPO_TIMESTEPS=30000 bash run.sh
```

Defaults: `DQN_EPISODES=200`, `PPO_TIMESTEPS=30000`. Each stage is also wall-clock-capped at 7 min so a misconfigured run can never block the pipeline.

---

## Manual usage

### Train + evaluate without the bash script

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-baselines.txt

python train/train_dqn.py --episodes 300
python train/train_ppo.py --timesteps 50000
python eval/evaluate.py --policies random greedy dqn ppo grpo_gguf
```

### Drive the model from the GUI

```bash
python run_gui.py --agent llm_grpo_gguf --gguf-n-gpu-layers -1
# or, from the terminal (no Pygame):
python run_gui.py --no-gui --agent llm_grpo_gguf --gguf-log-io
```

Press `R` to reset, `Esc` to quit (GUI). `--gguf-log-io` prints the raw model output and parsed action per step.

### Play yourself

```bash
python run_gui.py --agent manual
```

Click two adjacent candies to swap them. `N` does a random valid swap; `R` resets.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: numpy._core.numeric` | Loading a checkpoint pickled with NumPy 2.x on a NumPy 1.x install | `pip install 'numpy>=2.0'` (already pinned in `requirements-baselines.txt`) |
| `llama-cpp-python` builds from source for ages on aarch64 | No prebuilt wheel for that platform/Python | Make sure `build-essential` is installed; the build takes ~3 min and is cached |
| GRPO inference hangs at >1 min/move | CPU-only and `--gguf-n-gpu-layers 0` | Pass `--gguf-n-gpu-layers -1` if you have Metal/CUDA |
| GUI reports `pygame.error: No available video device` | Headless Linux with no `$DISPLAY` | Use `python run_gui.py --no-gui …` instead |

---

## License

MIT (see [`LICENSE`](LICENSE) if present in the repo root, otherwise inherit upstream).
