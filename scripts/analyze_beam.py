#!/usr/bin/env python3
"""Why wider beams hurt at our scale: brevity, precision, and length normalisation.

The paper reports BLEU rising monotonically with beam size (33.00 -> 34.50 ->
34.81 for its 5-model ensemble). Ours peaks at beam 2 and falls at beam 12.
This script decomposes BLEU into the brevity penalty and the n-gram precisions
for both the paper-faithful ranking (raw log-probability) and a
length-normalised ranking, so the cause is visible rather than asserted.

    python scripts/analyze_beam.py            # writes results/wmt14_small/beam_ablation.{md,json}
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO = Path(__file__).resolve().parents[1]


def work_dir() -> Path:
    return Path(os.environ.get("EE5180_WORK", Path.home() / "ee5180-work")).expanduser()


def score(hypPath: Path, refs: list[str]) -> dict | None:
    from sacrebleu.metrics import BLEU

    if not hypPath.exists():
        return None
    hyps = hypPath.read_text(encoding="utf-8").splitlines()
    scored = BLEU(tokenize="none", force=True).corpus_score(hyps, [refs])
    totalWords = sum(len(h.split()) for h in hyps)
    unkCount = sum(h.split().count("<unk>") for h in hyps)
    return {
        "bleu": round(scored.score, 2), "bp": round(scored.bp, 3),
        "len_ratio": round(scored.sys_len / scored.ref_len, 3),
        "p1": round(scored.precisions[0], 2), "p4": round(scored.precisions[3], 2),
        "words": totalWords,
        "unk": unkCount,
        "unk_pct": round(100 * unkCount / max(totalWords, 1), 1),
    }


def main() -> None:
    refs = (work_dir() / "data/wmt14/prepared/test.tok.fr").read_text(
        encoding="utf-8"
    ).splitlines()
    outDir = REPO / "results/wmt14_small"
    rows = []
    for label, folder, stem in (
        ("raw log-probability (paper's setting)", outDir / "hypotheses", "rev_seed1_beam{}"),
        ("length-normalised (alpha=1)", outDir / "lengthnorm", "rev_seed1_ln_beam{}"),
    ):
        for beam in (1, 2, 12):
            scored = score(folder / f"{stem.format(beam)}.hyp.tok.fr", refs)
            if scored:
                rows.append({"ranking": label, "beam": beam, **scored})

    if not rows:
        sys.exit("no hypothesis files found - run make_results first")
    (outDir / "beam_ablation.json").write_text(json.dumps(rows, indent=2))

    lines = ["## Why a wider beam hurts at our scale", "",
             "The paper's BLEU rises with beam size. Ours peaks at beam 2. Decomposing BLEU shows",
             "why, and shows that length normalisation is *not* the fix.", "",
             "| ranking | beam | BLEU | brevity penalty | length ratio | 1-gram prec | 4-gram prec |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['ranking']} | {row['beam']} | **{row['bleu']:.2f}** | {row['bp']:.3f} | "
                     f"{row['len_ratio']:.3f} | {row['p1']:.2f} | {row['p4']:.2f} |")

    raw = {r["beam"]: r for r in rows if r["ranking"].startswith("raw")}
    normalised = {r["beam"]: r for r in rows if r["ranking"].startswith("length")}
    lines += ["", "**Reading it.**", ""]
    if raw:
        lines.append(f"- With the paper's raw-log-probability ranking, widening the beam shortens the "
                     f"output: length ratio {raw[1]['len_ratio']:.3f} -> {raw[12]['len_ratio']:.3f} and the "
                     f"brevity penalty {raw[1]['bp']:.3f} -> {raw[12]['bp']:.3f}. Every extra token lowers a "
                     f"sequence's total log-probability, so a larger beam prefers shorter hypotheses.")
    if normalised and raw:
        lines.append(f"- Length normalisation removes the brevity penalty entirely (ratio "
                     f"{normalised[12]['len_ratio']:.3f}, BP {normalised[12]['bp']:.3f}) but "
                     f"**does not fix BLEU**: it overshoots into over-long output and 1-gram "
                     f"precision falls {raw[12]['p1']:.2f} -> {normalised[12]['p1']:.2f}, so "
                     f"BLEU drops further "
                     f"({raw[12]['bleu']:.2f} -> {normalised[12]['bleu']:.2f}).")
        lines.append("- So the cause is not calibration of length alone. A wider beam genuinely finds "
                 "*higher-probability* hypotheses -- 4-gram precision rises with beam under the raw "
                 "ranking -- but at this scale the model's likelihood and translation quality have "
                 "diverged, so searching harder optimises the wrong thing. This is the documented "
                 "'beam search curse'; the paper's model, trained on 24x more data, was accurate "
                 "enough not to suffer it.")
    if raw:
        lines.append(f"- Roughly **{raw[12]['unk_pct']:.0f}% of our output tokens are `<unk>`** "
                     f"({raw[12]['unk']:,} of {raw[12]['words']:,}), a direct consequence of the 32k "
                     f"vocabulary. The paper notes its own BLEU was penalised on out-of-vocabulary "
                     f"words at 80k; ours is penalised far harder, and that is a large part of the "
                     f"absolute gap.")
    (outDir / "beam_ablation.md").write_text("\n".join(lines) + "\n")
    print(f"[beam ablation] -> {outDir/'beam_ablation.md'} ({len(rows)} rows)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
