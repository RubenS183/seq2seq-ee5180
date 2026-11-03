#!/usr/bin/env bash
# Pull the trained weights and evaluation data from the GitHub release.
#
# Checkpoints are 220-229 MB each and GitHub rejects any file over 100 MB, so
# they cannot be in the repository. They are release assets instead, which do
# not count against repository size.
#
# Nothing REPORTED depends on this: every number in the report and the deck is
# committed under results/. This is for rerunning the three gates that need
# weights -- baseline-regression, probe, alignments -- from a clone.
#
#   ./scripts/fetch_artifacts.sh           # everything (892 MB)
#   ./scripts/fetch_artifacts.sh midterm   # just the mid-term pair (443 MB)
#   ./scripts/fetch_artifacts.sh endterm   # just the two end-term arms (453 MB)
set -euo pipefail
REPO_SLUG="${EE5180_REPO:-RubenS183/seq2seq-ee5180}"
TAG="${EE5180_RELEASE_TAG:-endterm-v1}"
WORK="${EE5180_WORK:-$HOME/ee5180-work}"
WHICH="${1:-all}"

command -v gh >/dev/null || {
  echo "needs the GitHub CLI (brew install gh), or download by hand from:"
  echo "  https://github.com/$REPO_SLUG/releases/tag/$TAG"
  exit 1
}

case "$WHICH" in
  all)      want=(runs-wmt14-fwd_seed1.tar runs-wmt14-rev_seed1.tar
                  runs-wmt14_endterm-rev_seed1.tar runs-wmt14_endterm-rev_attn_seed1.tar) ;;
  midterm)  want=(runs-wmt14-fwd_seed1.tar runs-wmt14-rev_seed1.tar) ;;
  endterm)  want=(runs-wmt14_endterm-rev_seed1.tar runs-wmt14_endterm-rev_attn_seed1.tar) ;;
  *) echo "usage: $0 [all|midterm|endterm]"; exit 2 ;;
esac
want+=(data-wmt14-eval-only.tar.gz SHA256SUMS.txt)

mkdir -p "$WORK"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "release $TAG -> $WORK"
for a in "${want[@]}"; do
  echo "  fetching $a"
  gh release download "$TAG" --repo "$REPO_SLUG" --pattern "$a" --dir "$STAGE" --clobber
done

# The manifest covers every asset in the release, so grep down to the ones we
# actually pulled before checking -- otherwise a partial fetch always "fails".
( cd "$STAGE"
  : > checked.txt
  for a in "${want[@]}"; do
    [ "$a" = "SHA256SUMS.txt" ] && continue
    grep "  $a\$" SHA256SUMS.txt >> checked.txt
  done
  shasum -a 256 -c checked.txt )

for a in "${want[@]}"; do
  case "$a" in
    *.tar)    tar -xf  "$STAGE/$a" -C "$WORK" ;;
    *.tar.gz) tar -xzf "$STAGE/$a" -C "$WORK" ;;
  esac
done

echo
echo "placed under $WORK:"
find "$WORK/runs" -name best.pt 2>/dev/null | sed "s|$WORK/|  |"
echo "  data/wmt14/prepared, data/wmt14/prepared_nofilter (eval splits only)"
echo
case "$WHICH" in
  all)     echo "now runnable:  make baseline-regression | make probe | make alignments" ;;
  midterm) echo "now runnable:  make baseline-regression" ;;
  endterm) echo "now runnable:  make probe | make alignments" ;;
esac
echo "retraining also needs the training split: make data / make data-endterm"
