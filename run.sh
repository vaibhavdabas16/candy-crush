#!/usr/bin/env bash
# End-to-end Candy Crush RL pipeline.
#
# Stages:
#   1. System setup (apt: python3, venv, build-essential, git, ca-certs)
#   2. Create .venv and install baseline + GRPO-inference deps
#   3. Train DQN  (~5 minutes wall-clock)
#   4. Train PPO  (~5 minutes wall-clock)
#   5. Run a terminal demo of all four baselines (random / greedy / DQN /
#      PPO) on the same seeds, using the freshly trained checkpoints, and
#      print a comparison table.
#   6. Download the merged Qwen GRPO Q4_K_M GGUF (~5.3 GB, one-time) and
#      evaluate it against greedy on the same seeds.
#   7. (END) Install GRPO training deps and attempt 1 hour of LoRA GRPO
#      training of the Qwen base model. This stage is gated to the end
#      because the previous stages are the actual deliverable; the hour
#      of training is a structural part of the pipeline only.
#
# Tested in a fresh ubuntu:22.04 docker container. Run with:
#   bash run.sh
#
# All paths are relative. All artifacts land under ./logs and ./models.
set -euo pipefail

cd "$(dirname "$0")"

# Allow the user to override seeds / max-moves from the command line:
#   bash run.sh --seeds 0 1 2 --max-moves 20
PASSED_ARGS=("$@")

SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
fi

mkdir -p logs models

banner() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

###############################################################################
# Stage 1: system setup
###############################################################################
banner "Stage 1/7  System setup"
need_apt_install=0
for cmd in python3 git cc; do
  command -v "$cmd" >/dev/null 2>&1 || need_apt_install=1
done
if ! python3 -c "import venv, ensurepip" >/dev/null 2>&1; then
  need_apt_install=1
fi
if [ "$need_apt_install" -eq 1 ] && command -v apt-get >/dev/null 2>&1; then
  echo "Installing python3, python3-venv, python3-pip, build-essential, git, ca-certificates"
  $SUDO apt-get update -y
  $SUDO apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip ca-certificates build-essential git
fi

###############################################################################
# Stage 2: venv + baseline + GRPO-inference deps
###############################################################################
banner "Stage 2/7  Python venv + baseline / GRPO-inference deps"
VENV=".venv"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"

python -m pip install --upgrade pip wheel
python -m pip install --no-cache-dir -r requirements-baselines.txt

###############################################################################
# Stage 3: Train DQN (~5 min)
###############################################################################
banner "Stage 3/7  Train DQN  (target ~5 min)"
DQN_EPISODES=${DQN_EPISODES:-200}
echo "Training DQN for ${DQN_EPISODES} episodes (max-moves 20)"
# Wall-clock cap of 7 minutes. The script saves models/dqn.pt at the end.
# If the cap fires we keep whatever model was already on disk.
timeout --preserve-status 420 python -u train/train_dqn.py \
  --episodes "$DQN_EPISODES" --max-moves 20 --log-every 25 \
  2>&1 | tee logs/train_dqn.log || {
    echo "DQN training was capped or failed - keeping existing models/dqn.pt"
}

###############################################################################
# Stage 4: Train PPO (~5 min)
###############################################################################
banner "Stage 4/7  Train PPO  (target ~5 min)"
PPO_TIMESTEPS=${PPO_TIMESTEPS:-30000}
echo "Training PPO for ${PPO_TIMESTEPS} timesteps"
timeout --preserve-status 420 python -u train/train_ppo.py \
  --timesteps "$PPO_TIMESTEPS" --max-moves 20 \
  2>&1 | tee logs/train_ppo.log || {
    echo "PPO training was capped or failed - keeping existing models/ppo.zip"
}

###############################################################################
# Stage 5: Baseline demo (random, greedy, DQN, PPO) on common seeds
###############################################################################
banner "Stage 5/7  Baseline terminal demo + comparison table"
python -u scripts/play_baselines_demo.py "${PASSED_ARGS[@]}" \
  2>&1 | tee logs/run_baselines.log

###############################################################################
# Stage 6: Evaluate the pre-trained Qwen GRPO GGUF on the same seeds
###############################################################################
banner "Stage 6/7  GRPO GGUF demo (downloads ~5.3 GB on first run)"
python -u scripts/play_grpo_demo.py "${PASSED_ARGS[@]}" \
  2>&1 | tee logs/run_grpo_eval.log

###############################################################################
# Stage 7 (END): 1 hour of GRPO LoRA training on the Qwen base model
###############################################################################
banner "Stage 7/7  GRPO LoRA training  (1 hour wall-clock cap)"
echo "Installing heavy GRPO training deps (transformers / peft / trl / datasets / accelerate)"
python -m pip install --no-cache-dir -r requirements-train-grpo.txt || {
  echo "Training deps did not install cleanly - skipping GRPO training stage."
  echo "==> Pipeline complete (stages 1-6)."
  exit 0
}

mkdir -p models/llm_grpo_candy
echo
echo "Starting GRPO training (capped at 3600 s)."
echo "On a CPU-only ubuntu:22.04 this stage will not converge - it"
echo "exists to demonstrate the training pipeline runs end-to-end. The"
echo "real trained adapter / GGUF used in stage 6 is hosted on HF at"
echo "  https://huggingface.co/arnavm7/candy-crush-qwen35-grpo-lora"
echo
timeout --preserve-status 3600 python -u train/train_llm_grpo_candy.py \
  --model-name Qwen/Qwen3.5-0.5B \
  --run-dir models/llm_grpo_candy \
  --output-dir models/llm_grpo_candy/run_local \
  --iterations 4 \
  --max-steps 4 \
  --train-prompts 16 \
  --num-generations 2 \
  --eval-episodes 2 \
  --skip-baseline-training \
  --no-4bit \
  2>&1 | tee logs/train_grpo.log || {
    echo "GRPO training stage hit the 1 h cap or failed - this is expected"
    echo "on CPU-only environments. The downloaded GGUF used in stage 6 is"
    echo "the canonical trained model."
}

banner "Pipeline complete"
echo "Artifacts:"
echo "   logs/train_dqn.log         DQN training log"
echo "   logs/train_ppo.log         PPO training log"
echo "   logs/run_baselines.log     Baseline demo transcript"
echo "   logs/run_grpo_eval.log     GRPO demo transcript"
echo "   logs/train_grpo.log        GRPO training log"
echo "   models/dqn.pt              Trained DQN policy"
echo "   models/ppo.zip             Trained PPO policy"
echo "   models/llm_grpo_candy/     GRPO training run output"
