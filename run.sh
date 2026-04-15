#!/usr/bin/env bash
# End-to-end demo for the four baseline Candy Crush agents.
#
#   - random   (no model)
#   - greedy   (no model)
#   - DQN      (uses models/dqn.pt - already in the repo)
#   - PPO      (uses models/ppo.zip - already in the repo)
#
# Plays a full 20-move episode for every agent on each seed in {0, 1, 2},
# prints the board + every action, and finishes with a comparison table.
# All artifacts (the full transcript and a CSV of rewards) land in ./logs/.
#
# Tested in a fresh ubuntu:22.04 docker container. Run with:
#   bash run.sh
#
# Override defaults from the command line, e.g.:
#   bash run.sh --seeds 10 11 12 --max-moves 30
set -euo pipefail

cd "$(dirname "$0")"

SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
fi

# --- system packages (Ubuntu 22.04 docker has apt + python3.10 already) ---
need_apt_install=0
if ! command -v python3 >/dev/null 2>&1; then
  need_apt_install=1
fi
if ! python3 -c "import venv, ensurepip" >/dev/null 2>&1; then
  need_apt_install=1
fi
if [ "$need_apt_install" -eq 1 ] && command -v apt-get >/dev/null 2>&1; then
  echo "==> Installing system packages: python3, python3-venv, python3-pip"
  $SUDO apt-get update -y
  $SUDO apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip ca-certificates
fi

# --- python venv ---
VENV=".venv"
if [ ! -d "$VENV" ]; then
  echo "==> Creating virtual environment at $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"

echo "==> Upgrading pip and installing baseline requirements"
python -m pip install --upgrade pip wheel
python -m pip install --no-cache-dir -r requirements-baselines.txt

# --- run demo ---
mkdir -p logs

echo "==> Running baseline demo (random, greedy, DQN, PPO)"
python -u scripts/play_baselines_demo.py "$@" 2>&1 | tee logs/run_baselines.log

echo
echo "==> Done."
echo "   Full transcript : logs/run_baselines.log"
