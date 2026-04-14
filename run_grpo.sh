#!/usr/bin/env bash
# End-to-end demo for the Qwen GRPO GGUF agent.
#
# This is the heavy script: each move calls a 9B-parameter LLM, so a
# 20-move episode takes ~30-60 seconds even on a CPU-only machine.
# Default seed list is {0, 1, 2} -> roughly 2-5 minutes of inference.
#
# What it does:
#   1. Creates a separate venv (.venv-grpo) with llama-cpp-python.
#   2. Downloads candy-crush-qwen35-grpo-Q4_K_M.gguf (~5.3 GB) from
#      huggingface.co/arnavm7/candy-crush-qwen35-grpo-lora into
#      models/llm_grpo_candy/qwen35_9b/gguf/  (skipped if already there).
#   3. Plays a full 20-move episode of the GRPO agent and the greedy
#      baseline on each seed, prints the board + every action, and
#      finishes with a comparison table.
#
# Tested in a fresh ubuntu:22.04 docker container. Run with:
#   bash run_grpo.sh
#
# Override defaults from the command line, e.g. (use GPU if you have one):
#   bash run_grpo.sh --seeds 0 1 --gguf-n-gpu-layers -1
set -euo pipefail

cd "$(dirname "$0")"

SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
fi

# --- system packages ---
need_apt_install=0
if ! command -v python3 >/dev/null 2>&1; then need_apt_install=1; fi
if ! python3 -c "import venv, ensurepip" >/dev/null 2>&1; then need_apt_install=1; fi
if ! command -v cc >/dev/null 2>&1; then need_apt_install=1; fi
if [ "$need_apt_install" -eq 1 ] && command -v apt-get >/dev/null 2>&1; then
  echo "==> Installing system packages: python3, python3-venv, python3-pip, build-essential, ca-certificates"
  $SUDO apt-get update -y
  $SUDO apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip ca-certificates build-essential
fi

# --- python venv (separate from baselines venv) ---
VENV=".venv-grpo"
if [ ! -d "$VENV" ]; then
  echo "==> Creating virtual environment at $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"

echo "==> Upgrading pip and installing GRPO requirements"
python -m pip install --upgrade pip wheel
# llama-cpp-python publishes prebuilt linux x86_64 wheels for recent
# Python versions; if no wheel is found pip will fall back to a source
# build, which is why build-essential is installed above.
python -m pip install --no-cache-dir -r requirements-grpo.txt

# --- run demo ---
mkdir -p logs models/llm_grpo_candy/qwen35_9b/gguf

echo "==> Running GRPO GGUF demo (this is the slow one - one inference per move)"
python -u scripts/play_grpo_demo.py "$@" 2>&1 | tee logs/run_grpo.log

echo
echo "==> Done."
echo "   Full transcript : logs/run_grpo.log"
echo "   Reward CSV      : logs/grpo_comparison.csv"
