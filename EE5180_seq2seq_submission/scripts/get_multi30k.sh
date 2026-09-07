#!/usr/bin/env bash
# Tier 0 data: Multi30k task1, English -> French (29k train / 1014 val / 1000 test2016).
# Same language direction as the paper, small enough to validate the pipeline in minutes.
set -euo pipefail
WORK="${EE5180_WORK:-$HOME/ee5180-work}"
OUT="$WORK/data/multi30k/raw"
BASE="https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw"
mkdir -p "$OUT"
for split in train val test_2016_flickr; do
  for lang in en fr; do
    if [ ! -s "$OUT/$split.$lang" ]; then
      echo "[get] $split.$lang"
      curl -fsSL --retry 3 -C - -o "$OUT/$split.$lang.gz" "$BASE/$split.$lang.gz"
      gunzip -f "$OUT/$split.$lang.gz"
    fi
  done
done
wc -l "$OUT"/*.en "$OUT"/*.fr
