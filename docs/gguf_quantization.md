# Q4_K_M GGUF Quantization

The released LoRA adapter can be run through the normal Transformers/PEFT path, but CPU and Mac inference are much more practical with a merged GGUF model.

This repo exports:

```text
arnavm7/candy-crush-qwen35-grpo-lora/gguf/candy-crush-qwen35-grpo-Q4_K_M.gguf
```

The GGUF is a merged model: `Qwen/Qwen3.5-9B` plus the trained Candy Crush GRPO LoRA adapter. It does not require loading the LoRA adapter separately.

## Artifact

| File | Value |
| --- | --- |
| Quantization | `Q4_K_M` |
| Size | 5.3 GB |
| Quantized tensor size from llama.cpp | 5357.88 MiB |
| SHA256 | `96b9c951c8f76a9b8b18f8c593871ee6c270e05adac6eb2fd10a1bd3c37b90cc` |
| Smoke-test output | `swap (0, 3) (0, 4)` |

Smoke test used llama.cpp `llama-completion` with `-c 4096`, `-t 16`, `-n 24`, and `--temp 0`. On the Linux CPU test machine, the measured prompt rate was about 31.5 tokens/s and generation was about 3.2 tokens/s for the full Candy Crush prompt.

## Download

```bash
pip install -U huggingface_hub

huggingface-cli download arnavm7/candy-crush-qwen35-grpo-lora \
  gguf/candy-crush-qwen35-grpo-Q4_K_M.gguf \
  gguf/candy-crush-qwen35-grpo-Q4_K_M.gguf.sha256 \
  --local-dir models/llm_grpo_candy/qwen35_9b

sha256sum -c models/llm_grpo_candy/qwen35_9b/gguf/candy-crush-qwen35-grpo-Q4_K_M.gguf.sha256
```

On macOS, use `shasum -a 256` if `sha256sum` is not installed.

## Run With llama.cpp

Build a recent llama.cpp from source if your package manager version cannot load `general.architecture = qwen35`.

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --target llama-completion -j "$(nproc)"
```

Run a direct completion:

```bash
PROMPT='Task: choose one legal Candy Crush swap.
Coordinates are zero-indexed as (row,col).
Required first-line format:
swap (r,c) (r,c)
Board:
0 1 2 3 4 5 0 1
1 2 3 4 5 0 1 2
2 3 4 5 0 1 2 3
3 4 5 0 1 2 3 4
4 5 0 1 2 3 4 5
5 0 1 2 3 4 5 0
0 1 2 3 4 5 0 1
1 2 3 4 5 0 1 2
Answer now. First line only the swap command:'

llama.cpp/build/bin/llama-completion \
  -m models/llm_grpo_candy/qwen35_9b/gguf/candy-crush-qwen35-grpo-Q4_K_M.gguf \
  -c 4096 \
  -t 8 \
  -n 24 \
  --temp 0 \
  --no-display-prompt \
  --no-conversation \
  --single-turn \
  --simple-io \
  -p "$PROMPT"
```

For real board quality, use the same prompt format as `train/train_llm_grpo_candy.py::prompt_for_env`, including the current board, special candy rules, and valid action list.

## Recreate The GGUF

Merge the LoRA adapter into the base model:

```bash
python scripts/merge_qwen_grpo_lora.py \
  --base-model Qwen/Qwen3.5-9B \
  --adapter arnavm7/candy-crush-qwen35-grpo-lora \
  --output-dir models/llm_grpo_candy/qwen35_9b/merged_hf \
  --dtype bfloat16
```

Convert the merged Hugging Face model to BF16 GGUF:

```bash
python /path/to/llama.cpp/convert_hf_to_gguf.py \
  models/llm_grpo_candy/qwen35_9b/merged_hf \
  --outfile models/llm_grpo_candy/qwen35_9b/gguf/candy-crush-qwen35-grpo-bf16.gguf \
  --outtype bf16 \
  --model-name candy-crush-qwen35-grpo
```

Quantize to Q4_K_M:

```bash
/path/to/llama.cpp/build/bin/llama-quantize \
  models/llm_grpo_candy/qwen35_9b/gguf/candy-crush-qwen35-grpo-bf16.gguf \
  models/llm_grpo_candy/qwen35_9b/gguf/candy-crush-qwen35-grpo-Q4_K_M.gguf \
  Q4_K_M
```

Then record the checksum:

```bash
sha256sum models/llm_grpo_candy/qwen35_9b/gguf/candy-crush-qwen35-grpo-Q4_K_M.gguf \
  > models/llm_grpo_candy/qwen35_9b/gguf/candy-crush-qwen35-grpo-Q4_K_M.gguf.sha256
```

If conversion fails with an unrecognized Qwen3.5 tokenizer pre-tokenizer hash, use a newer llama.cpp build. The local conversion used Qwen3.5 architecture support and mapped tokenizer hash `1444df51289cfa8063b96f0e62b1125440111bc79a52003ea14b6eac7016fd5f` to `qwen35`.

## GUI Status

The current Pygame GUI agent loads the Hugging Face adapter through Transformers/PEFT. The GGUF path is for direct llama.cpp inference and efficient Mac/CPU testing. To use GGUF inside the GUI, add a llama.cpp-backed `llm_grpo` implementation, then pass it the same board prompt produced by the existing agent.
