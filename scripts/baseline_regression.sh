#!/usr/bin/env bash
# Did adding attention change the paper's model? Byte-exact answer.
#
# The end-term refactor touched Encoder.forward (it now also returns the
# per-position annotations) and beam.py (survivor reindexing goes through
# DecodeState). Neither should alter the no-attention arm by a single float --
# but "BLEU still rounds to 6.60" is a weak way to check that.
#
# So: re-decode the committed mid-term checkpoints and `diff` the hypotheses
# against the files that were submitted. Decoding is deterministic (CPU, no_grad,
# no dropout), so the diff must be EMPTY, not small.
#
#   ./scripts/baseline_regression.sh            # full 3003 sentences, ~19 min
#   LIMIT=200 ./scripts/baseline_regression.sh  # ~80 s, for a dev loop
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${EE5180_WORK:-$HOME/ee5180-work}"
PY="${EE5180_PY:-$WORK/.venv/bin/python}"
LIMIT="${LIMIT:-0}"
BEAM="${BEAM:-12}"
OUT="$(mktemp -d)"
cd "$REPO"

fail=0
for dir in rev fwd; do
  flag="--reverse-source"; [ "$dir" = "fwd" ] && flag=""
  ref="results/wmt14_small/hypotheses/${dir}_seed1_beam${BEAM}.hyp.tok.fr"
  [ -f "$ref" ] || { echo "SKIP $dir: no committed reference at $ref"; continue; }
  PYTHONPATH=src "$PY" -m seq2seq.evaluate \
    --checkpoint "$WORK/runs/wmt14/${dir}_seed1/best.pt" \
    --data-dir "$WORK/data/wmt14/prepared" \
    --beam "$BEAM" $flag --limit "$LIMIT" \
    --out-dir "$OUT" --tag "${dir}_seed1" >/dev/null 2>&1
  got="$OUT/${dir}_seed1_beam${BEAM}.hyp.tok.fr"
  if [ "$LIMIT" != "0" ]; then
    # compare only the decoded prefix
    head -n "$LIMIT" "$ref" > "$OUT/ref.$dir"
  else
    cp "$ref" "$OUT/ref.$dir"
  fi
  if diff -q "$OUT/ref.$dir" "$got" >/dev/null 2>&1; then
    bleu=$("$PY" -c "import json;print(json.load(open('$OUT/${dir}_seed1_beam${BEAM}.eval.json'))['bleu_tok'])")
    echo "  PASS $dir: hypotheses byte-identical to the submitted mid-term (BLEU $bleu)"
  else
    n=$(diff "$OUT/ref.$dir" "$got" | grep -c '^[<>]' || true)
    echo "  FAIL $dir: $n differing lines -- the refactor changed the baseline"
    diff "$OUT/ref.$dir" "$got" | head -6
    fail=1
  fi
done
rm -rf "$OUT"
[ "$fail" = 0 ] && echo "baseline regression: PASS" || echo "baseline regression: FAIL"
exit "$fail"
