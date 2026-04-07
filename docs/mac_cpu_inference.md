# MacBook And CPU Inference

This guide explains how to run the trained Candy Crush Qwen GRPO LoRA adapter directly on a MacBook or CPU-only machine.

For efficient non-CUDA inference, prefer the merged `Q4_K_M` GGUF artifact:

```text
arnavm7/candy-crush-qwen35-grpo-lora/gguf/candy-crush-qwen35-grpo-Q4_K_M.gguf
```

That file is 5.3 GB, runs with llama.cpp, and does not need the Qwen base model or LoRA adapter to be loaded separately. Full commands are in [gguf_quantization.md](gguf_quantization.md).

The trained adapter is:

```text
arnavm7/candy-crush-qwen35-grpo-lora
```

It is a LoRA adapter for:

```text
Qwen/Qwen3.5-9B
```

That means inference must load the Qwen 9B base model plus the LoRA adapter. The adapter is small; the base model is the expensive part.

## Practical Hardware Expectations

| Machine | Expected result |
| --- | --- |
| NVIDIA GPU | Best path. Use CUDA 4-bit loading. |
| Apple Silicon MacBook, 16 GB unified memory | Usually too tight for Qwen 9B PyTorch inference. It may OOM or heavily swap. |
| Apple Silicon MacBook, 32 GB unified memory | Possible with MPS fp16, but close other apps and expect slow moves. |
| Apple Silicon MacBook, 64 GB+ unified memory | Recommended Mac path. |
| CPU-only, 32 GB RAM | Usually too tight or impractically slow. |
| CPU-only, 64 GB+ RAM | Use the Q4_K_M GGUF with llama.cpp if possible. Transformers CPU is a compatibility path only. |

For Mac and CPU, do not use `bitsandbytes` 4-bit. `bitsandbytes` is for CUDA/NVIDIA in this project. Use the GGUF for efficiency, or use `--llm-no-4bit` for the Transformers/PEFT path.

## Install On Apple Silicon Mac

Use Python 3.10 or 3.11.

```bash
git clone -b llm-grpo-qwen-candy https://github.com/vaibhavdabas16/candy-crush.git
cd candy-crush

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

pip install torch torchvision torchaudio
pip install gymnasium numpy stable-baselines3 sb3-contrib tensorboard pygame
pip install transformers peft accelerate datasets trl huggingface_hub safetensors
```

This intentionally does not install `bitsandbytes`.

## Run The GUI On MacBook

Fastest simple command:

```bash
python run_gui.py \
  --agent llm_grpo \
  --llm-grpo-path arnavm7/candy-crush-qwen35-grpo-lora \
  --llm-model-name Qwen/Qwen3.5-9B \
  --llm-device mps \
  --llm-no-4bit \
  --llm-dtype float16 \
  --llm-max-new-tokens 16 \
  --agent-delay 2.0
```

The first run downloads the base model and adapter from Hugging Face. Later runs use the local Hugging Face cache.

If MPS runs out of memory, try:

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python run_gui.py \
  --agent llm_grpo \
  --llm-grpo-path arnavm7/candy-crush-qwen35-grpo-lora \
  --llm-model-name Qwen/Qwen3.5-9B \
  --llm-device mps \
  --llm-no-4bit \
  --llm-dtype float16 \
  --llm-max-new-tokens 16 \
  --agent-delay 3.0
```

If it still fails, the machine probably does not have enough unified memory for Qwen 9B in this runtime.

## Direct Non-GUI Inference

The Hugging Face model repo also includes a standalone inference package. Use this when you want to run the trained model directly and get a JSON swap recommendation without the Pygame GUI.

```bash
git clone https://huggingface.co/arnavm7/candy-crush-qwen35-grpo-lora
cd candy-crush-qwen35-grpo-lora/standalone

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

On Apple Silicon:

```bash
PYTHONPATH=src python -m candy_grpo.recommend \
  --adapter-id arnavm7/candy-crush-qwen35-grpo-lora \
  --base-model Qwen/Qwen3.5-9B \
  --device mps \
  --no-4bit \
  --dtype float16 \
  --max-new-tokens 16 \
  --json
```

On CPU:

```bash
PYTHONPATH=src python -m candy_grpo.recommend \
  --adapter-id arnavm7/candy-crush-qwen35-grpo-lora \
  --base-model Qwen/Qwen3.5-9B \
  --device cpu \
  --no-4bit \
  --dtype float32 \
  --max-new-tokens 16 \
  --json
```

The JSON result contains a canonical `move` field, for example:

```json
[
  {
    "move": "swap (5,2) (5,3)",
    "legal": true,
    "immediate_reward": 140.0
  }
]
```

## Local Adapter Cache

For repeat runs, you can explicitly download the adapter into the repo. The base model will still come from the Hugging Face cache.

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="arnavm7/candy-crush-qwen35-grpo-lora",
    local_dir="models/llm_grpo_candy/qwen35_9b/final_plus30",
    ignore_patterns=["standalone/*"],
)
PY
```

Then run:

```bash
python run_gui.py \
  --agent llm_grpo \
  --llm-grpo-path models/llm_grpo_candy/qwen35_9b/final_plus30 \
  --llm-model-name Qwen/Qwen3.5-9B \
  --llm-device mps \
  --llm-no-4bit \
  --llm-dtype float16 \
  --llm-max-new-tokens 16 \
  --agent-delay 2.0
```

## CPU-only Run

CPU-only inference works the same way, but it is much slower.

Measured result from the Linux test machine used for this repo: with CUDA hidden, `--device cpu --no-4bit --dtype float32 --max-new-tokens 8` did not return one JSON recommendation within 10 minutes, so the run was stopped. That machine had enough RAM, but CPU-only Qwen 9B startup/generation was still not practical for interactive use.

Install without `bitsandbytes`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch
pip install gymnasium numpy stable-baselines3 sb3-contrib tensorboard pygame
pip install transformers peft accelerate datasets trl huggingface_hub safetensors
```

Run:

```bash
python run_gui.py \
  --agent llm_grpo \
  --llm-grpo-path arnavm7/candy-crush-qwen35-grpo-lora \
  --llm-model-name Qwen/Qwen3.5-9B \
  --llm-device cpu \
  --llm-no-4bit \
  --llm-dtype float32 \
  --llm-max-new-tokens 16 \
  --agent-delay 10.0
```

Use this mainly to validate that the code path can load on machines without CUDA. For actual gameplay or GUI usage, use CUDA or Apple Silicon MPS.

On some modern CPUs, `bfloat16` can reduce memory pressure:

```bash
python run_gui.py \
  --agent llm_grpo \
  --llm-grpo-path arnavm7/candy-crush-qwen35-grpo-lora \
  --llm-model-name Qwen/Qwen3.5-9B \
  --llm-device cpu \
  --llm-no-4bit \
  --llm-dtype bfloat16 \
  --llm-max-new-tokens 16 \
  --agent-delay 10.0
```

If `bfloat16` errors or produces unsupported-op failures, use `float32`.

## Why These Flags Matter

| Flag | Why it matters |
| --- | --- |
| `--llm-no-4bit` | Disables CUDA-only 4-bit loading. Required for Mac/CPU. |
| `--llm-device mps` | Uses Apple Silicon GPU acceleration. |
| `--llm-device cpu` | Forces CPU inference. |
| `--llm-dtype float16` | Best Mac memory/speed tradeoff for MPS. |
| `--llm-dtype float32` | Most compatible CPU dtype, but largest memory use. |
| `--llm-max-new-tokens 16` | Keeps generation short. The model only needs to output one swap. |
| `--agent-delay` | Gives the model enough time between GUI moves. Increase it on slow machines. |

## What To Expect

The model does not need to generate a long answer. The useful output is the first parseable command:

```text
swap (r,c) (r,c)
```

The GUI validates that command against `CandyEnv`. If the model output is invalid or cannot be parsed, the agent falls back to the best immediate legal move so the GUI keeps running.

For the most efficient non-CUDA experience, use:

```bash
--llm-device mps --llm-no-4bit --llm-dtype float16 --llm-max-new-tokens 16
```

For CPU-only testing, use:

```bash
--llm-device cpu --llm-no-4bit --llm-dtype float32 --llm-max-new-tokens 16
```
