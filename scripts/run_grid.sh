#!/usr/bin/env bash
# Train the {forward, reversed} x seeds grid for one config, sequentially.
# Sequential on purpose: two MPS processes contend for the same GPU and make
# the per-run timings meaningless.
set -euo pipefail
CONFIG="${1:?usage: run_grid.sh <config.yaml> [seeds...]}"; shift
SEEDS=("${@:-1}")
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Python to use: EE5180_PY if set, else the local venv, else whatever python3 is
# on PATH (which is the case on Colab/Kaggle).
if [ -n "${EE5180_PY:-}" ]; then PY="$EE5180_PY"
elif [ -x "$HOME/ee5180-work/.venv/bin/python" ]; then PY="$HOME/ee5180-work/.venv/bin/python"
else PY="$(command -v python3)"; fi
cd "$REPO"
for seed in "${SEEDS[@]}"; do
  for dir in rev fwd; do
    flag="--reverse-source"; [ "$dir" = "fwd" ] && flag="--forward-source"
    echo "=== $(basename "$CONFIG" .yaml) | $dir | seed $seed ==="
    PYTHONPATH=src "$PY" -m seq2seq.train --config "$CONFIG" $flag --seed "$seed"
  done
done
