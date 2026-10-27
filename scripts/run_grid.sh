#!/usr/bin/env bash
# Train a grid for one config, sequentially.
#
#   run_grid.sh <config.yaml> [seeds...]
#
# Sequential on purpose: two MPS processes contend for the same GPU and make the
# per-run timings meaningless.
#
# Two optional environment overrides, which is how the end-term adds its second
# dimension without changing the mid-term's invocations:
#
#   DIRS  source directions to train     (default "rev fwd")
#   ARMS  end-term arms to train         (default "base"; "base attn")
#
# `base` passes --attention none, `attn` passes --attention bahdanau. Run dirs
# are <run_dir>/{rev,fwd}_seed<N> for base and {rev,fwd}_attn_seed<N> for attn,
# so the mid-term layout is unchanged and the two arms never collide.
#
#   make smoke        -> DIRS="rev fwd" ARMS="base"          (mid-term, Tier 0)
#   make endterm-wmt  -> DIRS="rev"     ARMS="base attn"
set -euo pipefail
CONFIG="${1:?usage: run_grid.sh <config.yaml> [seeds...]}"; shift
SEEDS=("${@:-1}")
DIRS="${DIRS:-rev fwd}"
ARMS="${ARMS:-base}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Python to use: EE5180_PY if set, else the local venv, else whatever python3 is
# on PATH (which is the case on Colab/Kaggle).
if [ -n "${EE5180_PY:-}" ]; then PY="$EE5180_PY"
elif [ -x "$HOME/ee5180-work/.venv/bin/python" ]; then PY="$HOME/ee5180-work/.venv/bin/python"
else PY="$(command -v python3)"; fi
cd "$REPO"
for seed in "${SEEDS[@]}"; do
  for arm in $ARMS; do
    case "$arm" in
      base) attn="none" ;;
      attn) attn="bahdanau" ;;
      *) echo "unknown arm '$arm' (expected base|attn)" >&2; exit 2 ;;
    esac
    for dir in $DIRS; do
      flag="--reverse-source"; [ "$dir" = "fwd" ] && flag="--forward-source"
      echo "=== $(basename "$CONFIG" .yaml) | $dir | $arm | seed $seed ==="
      PYTHONPATH=src "$PY" -m seq2seq.train --config "$CONFIG" $flag \
        --attention "$attn" --seed "$seed"
    done
  done
done
