#!/usr/bin/env bash
# Tier 1 data: WMT'14 English -> French.
#
# Training corpora: News-Commentary v9 + Europarl v7, both constituent corpora
# of the official WMT'14 En-Fr training set. (The paper trained on a 12M-pair
# "selected" subset of the full WMT'14 data; we use ~2.2M pairs from these two,
# then subsample -- a declared deviation.)
#
# Dev  = newstest2013  (the paper used ntst12+13 for development)
# Test = newstest2014, the FULL 3003-sentence version = the paper's ntst14.
#        Fetched through sacrebleu so the exact official set is used.
set -euo pipefail
WORK="${EE5180_WORK:-$HOME/ee5180-work}"
# Python to use: EE5180_PY if set, else the local venv, else whatever python3 is
# on PATH (which is the case on Colab/Kaggle).
if [ -n "${EE5180_PY:-}" ]; then PY="$EE5180_PY"
elif [ -x "$HOME/ee5180-work/.venv/bin/python" ]; then PY="$HOME/ee5180-work/.venv/bin/python"
else PY="$(command -v python3)"; fi
RAW="$WORK/data/wmt14/raw"; DL="$WORK/data/wmt14/download"
mkdir -p "$RAW" "$DL"

fetch() { # url dest
  [ -s "$2" ] && { echo "[skip] $(basename "$2")"; return; }
  echo "[get] $(basename "$2")"
  curl -fL --retry 5 --retry-delay 3 -C - -o "$2" "$1"
}

fetch "https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz" "$DL/nc-v9.tgz"
fetch "https://www.statmt.org/europarl/v7/fr-en.tgz"             "$DL/europarl-fr-en.tgz"

echo "[extract]"
tar -xzf "$DL/nc-v9.tgz" -C "$DL" --strip-components=1 \
    training/news-commentary-v9.fr-en.en training/news-commentary-v9.fr-en.fr 2>/dev/null || \
  tar -xzf "$DL/nc-v9.tgz" -C "$DL"
tar -xzf "$DL/europarl-fr-en.tgz" -C "$DL"

find "$DL" -name 'news-commentary-v9.fr-en.en' -exec cp {} "$RAW/nc9.en" \;
find "$DL" -name 'news-commentary-v9.fr-en.fr' -exec cp {} "$RAW/nc9.fr" \;
find "$DL" -name 'europarl-v7.fr-en.en'        -exec cp {} "$RAW/europarl.en" \;
find "$DL" -name 'europarl-v7.fr-en.fr'        -exec cp {} "$RAW/europarl.fr" \;

echo "[dev/test via sacrebleu]"
"$PY" -m sacrebleu -t wmt13      -l en-fr --echo src > "$RAW/newstest2013.en"
"$PY" -m sacrebleu -t wmt13      -l en-fr --echo ref > "$RAW/newstest2013.fr"
"$PY" -m sacrebleu -t wmt14/full -l en-fr --echo src > "$RAW/newstest2014.en"
"$PY" -m sacrebleu -t wmt14/full -l en-fr --echo ref > "$RAW/newstest2014.fr"

wc -l "$RAW"/*.en "$RAW"/*.fr
echo "[done] newstest2014 should be 3003 lines (the paper's ntst14)"
