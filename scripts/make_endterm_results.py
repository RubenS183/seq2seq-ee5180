#!/usr/bin/env python3
"""Join the two end-term arms into the bottleneck result.

`make_results.py` produces one arm's table at a time. The end-term's claim is
about the DIFFERENCE between arms, so this script reads both and emits:

  bottleneck_by_length.png   both arms' BLEU-vs-source-length on one axis
  arm_comparison.png         the 2x2 grid as bars, with a seed spread if present
  endterm_results.md         the comparison table and the claims, stated with
                             their scale caveat
  endterm_results.json       every row that went into the above

The headline is deliberately NOT "attention scores higher". It is the SHAPE of
the length curve: a fixed-length vector has to carry a whole sentence, so it
should degrade as sentences grow and attention should not. A model that simply
scored a couple of BLEU higher everywhere would be a better model, not evidence
about a bottleneck.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seq2seq.plots import (  # noqa: E402
    plot_arm_bars,
    plot_bleu_by_length_overlay,
)

ARM_LABEL = {"base": "fixed vector", "attn": "+ attention"}


def load_rows(path: Path, arm: str) -> list[dict]:
    if not path.exists():
        print(f"[warn] missing {path}", file=sys.stderr)
        return []
    rows = json.loads(path.read_text())
    for row in rows:
        row.setdefault("arm", arm)
    return rows


def singles(rows, beam=None):
    singleModel = [row for row in rows if row["n_models"] == 1]
    return [row for row in singleModel if beam is None or row["beam_size"] == beam]


def dedupe(rows):
    """Drop rows that re-score an identical configuration.

    `make_results.py --per-seed` decodes the primary seed twice at the largest
    beam, once as that direction's single-model row and once in the per-seed
    sweep. Both rows are the same checkpoint at the same beam, so averaging over
    them weights the primary seed double and reports an inflated run count (3
    seeds shown as "4 runs"). Every statistic below is a mean or a standard
    deviation over these rows, so the duplicate has to go before any of them are
    computed.

    Keyed on the full configuration rather than the score, so two seeds that
    happen to tie are still two independent runs.
    """
    seen, kept = set(), []
    for row in rows:
        key = (row["direction"], row["arm"], row["beam_size"],
               tuple(row["seeds"]), row["n_models"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--base", required=True,
                        help="results dir of the no-attention arm")
    parser.add_argument("--attn", required=True,
                        help="results dir of the attention arm")
    parser.add_argument("--out", required=True)
    parser.add_argument("--beam", type=int, default=12,
                        help="beam for the headline table")
    parser.add_argument("--scale-note", default="")
    args = parser.parse_args()

    outDir = Path(args.out)
    outDir.mkdir(parents=True, exist_ok=True)
    rows = (load_rows(Path(args.base) / "all_results.json", "base")
            + load_rows(Path(args.attn) / "all_results.json", "attn"))
    if not rows:
        sys.exit("no result rows found in either arm; run make_results.py first")
    rawCount = len(rows)
    rows = dedupe(rows)
    if len(rows) != rawCount:
        print(f"[dedupe] dropped {rawCount - len(rows)} re-scored duplicate row(s) "
              f"of {rawCount}", file=sys.stderr)
    (outDir / "endterm_results.json").write_text(json.dumps(rows, indent=2))

    # The headline figure: both arms' length curves on one axis, one curve per
    # (direction, arm). Where several seeds exist, take the first.
    headline = [row for row in singles(rows, args.beam) if len(row["seeds"]) == 1]
    seen, curves = set(), []
    for row in headline:
        key = (row["direction"], row["arm"])
        if key not in seen:
            seen.add(key)
            curves.append(row)
    plot_bleu_by_length_overlay(
        curves, outDir / "bottleneck_by_length.png",
        f"BLEU by source length, fixed vector vs attention ({args.name})",
    )

    # The 2x2 grid as bars, with a seed spread where we have one.
    bars = []
    for direction in ("fwd", "rev"):
        for arm in ("base", "attn"):
            scores = [row["bleu_tok"] for row in singles(rows, args.beam)
                      if row["direction"] == direction and row["arm"] == arm
                      and len(row["seeds"]) == 1]
            if not scores:
                continue
            bars.append({
                "label": f"{'reversed' if direction == 'rev' else 'forward'}"
                         f"\n{ARM_LABEL[arm]}",
                "mean": statistics.fmean(scores),
                "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "n": len(scores),
                "reverse": direction == "rev",
                "arm": arm,
                "direction": direction,
            })
    if bars:
        plot_arm_bars(bars, outDir / "arm_comparison.png",
                      f"BLEU at beam {args.beam} ({args.name})")

    lines = [f"# End-term results - {args.name}", ""]
    if args.scale_note:
        lines += [f"> **Scale gap.** {args.scale_note}", ""]
    lines += [
        "The two arms share their corpus, vocabulary, seed, batch order, optimiser,",
        "learning-rate schedule and gradient clipping. `attention: bahdanau` is the",
        "only difference between them.", "",
        "| direction | arm | beam | BLEU (tok, cased) | BLEU (detok) | test ppl "
        "| params | runs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for bar in bars:
        matching = [row for row in singles(rows, args.beam)
                    if row["direction"] == bar["direction"] and row["arm"] == bar["arm"]
                    and len(row["seeds"]) == 1]
        meanDetok = statistics.fmean([row["bleu_detok"] for row in matching])
        perplexities = [row["test_ppl"] for row in matching if row.get("test_ppl")]
        params = matching[0].get("n_params")
        spread = f" ± {bar['std']:.2f}" if bar["n"] > 1 else ""
        direction = "reversed" if bar["reverse"] else "forward"
        meanPpl = f"{statistics.fmean(perplexities):.3f}" if perplexities else "-"
        lines.append(
            f"| {direction} | {ARM_LABEL[bar['arm']]} | {args.beam} | "
            f"**{bar['mean']:.2f}**{spread} | {meanDetok:.2f} | {meanPpl} | "
            f"{params/1e6:.1f}M | {bar['n']} |"
        )

    byArm = {(bar["direction"], bar["arm"]): bar for bar in bars}
    lines += ["", "## What the arms say about the bottleneck", ""]

    for direction in ("rev", "fwd"):
        if (direction, "base") in byArm and (direction, "attn") in byArm:
            baseMean = byArm[(direction, "base")]["mean"]
            attnMean = byArm[(direction, "attn")]["mean"]
            name = "reversed" if direction == "rev" else "forward"
            lines.append(f"- {name}: fixed vector **{baseMean:.2f}** -> attention "
                         f"**{attnMean:.2f}** (**{attnMean - baseMean:+.2f}** BLEU).")

    # Does attention make the reversal trick unnecessary? The paper reverses the
    # source to shorten the lag between a source word and the step that needs
    # it, so if the bottleneck is what reversal was working around, the gap
    # should shrink once attention exists.
    needed = (("fwd", "base"), ("rev", "base"), ("fwd", "attn"), ("rev", "attn"))
    if all(key in byArm for key in needed):
        baseGap = byArm[("rev", "base")]["mean"] - byArm[("fwd", "base")]["mean"]
        attnGap = byArm[("rev", "attn")]["mean"] - byArm[("fwd", "attn")]["mean"]
        lines += [
            "",
            f"- The reversal gap (reversed minus forward) is **{baseGap:+.2f}** without "
            f"attention and **{attnGap:+.2f}** with it.",
        ]
        if abs(attnGap) < abs(baseGap):
            absorbed = (1 - abs(attnGap) / abs(baseGap)) * 100 if baseGap else 0.0
            lines.append(
                f"  Attention absorbs {absorbed:.0f}% of it. That is the paper's trick "
                "being made redundant: reversing the source shortens the distance "
                "between a source word and the decoder step that needs it, and "
                "attention removes the distance altogether."
            )
        else:
            lines.append(
                "  The gap did **not** shrink at this scale. Reported rather than "
                "hidden: it is evidence against the strong reading of the bottleneck "
                "story, and the length curves below are the better-powered test."
            )

    lines += ["", "### The shape of the length curve", "",
              "This is the actual bottleneck claim: a single fixed vector should lose "
              "ground",
              "as the source grows, and attention should not.", ""]
    reversedCurves = {row["arm"]: row for row in curves if row["direction"] == "rev"}
    if "base" in reversedCurves and "attn" in reversedCurves:
        baseBuckets = reversedCurves["base"].get("bleu_by_length", [])
        attnBuckets = reversedCurves["attn"].get("bleu_by_length", [])
        baseBleu = {b["bucket"]: b["bleu_tok"] for b in baseBuckets}
        attnBleu = {b["bucket"]: b["bleu_tok"] for b in attnBuckets}
        # Sentence counts per bucket, so a trend read off a near-empty tail
        # bucket cannot be mistaken for a result.
        bucketCounts = {b["bucket"]: b.get("n") for b in baseBuckets}
        lines += ["| source length | sentences | fixed vector | + attention | gain |",
                  "|---|---|---|---|---|"]
        for bucket in [b["bucket"] for b in baseBuckets]:
            if bucket in attnBleu:
                count = bucketCounts.get(bucket)
                lines.append(f"| {bucket} | {count if count is not None else '-'} | "
                             f"{baseBleu[bucket]:.2f} | {attnBleu[bucket]:.2f} | "
                             f"**{attnBleu[bucket] - baseBleu[bucket]:+.2f}** |")
        shared = [b for b in baseBleu if b in attnBleu]
        if len(shared) >= 2:
            first, last = shared[0], shared[-1]
            baseDrop = baseBleu[last] - baseBleu[first]
            attnDrop = attnBleu[last] - attnBleu[first]
            firstGain = attnBleu[first] - baseBleu[first]
            lastGain = attnBleu[last] - baseBleu[last]
            lines += [
                "",
                f"- Fixed vector, {first} -> {last}: {baseBleu[first]:.2f} -> "
                f"{baseBleu[last]:.2f} ({baseDrop:+.2f}).",
                f"- Attention,    {first} -> {last}: {attnBleu[first]:.2f} -> "
                f"{attnBleu[last]:.2f} ({attnDrop:+.2f}).",
                "",
            ]
            # State which case the numbers actually show. The claim is about the
            # GAIN widening with length: attention is supposed to help most
            # where a fixed vector has most to carry.
            if lastGain > firstGain:
                lines.append(
                    f"The gain widens with source length ({firstGain:+.2f} at {first} -> "
                    f"{lastGain:+.2f} at {last}), which is the bottleneck claim: the "
                    f"longer the source, the more a single fixed vector costs."
                )
            else:
                lines.append(
                    f"**The gain does NOT widen with source length** ({firstGain:+.2f} at "
                    f"{first} -> {lastGain:+.2f} at {last}), and the attention arm falls "
                    f"off at least as fast as the fixed-vector arm ({attnDrop:+.2f} vs "
                    f"{baseDrop:+.2f}). At this scale and on this corpus the length curve "
                    f"does NOT evidence the bottleneck; the arm-level gain above is real, "
                    f"but it is not explained by source length."
                )
            tailCount = bucketCounts.get(last)
            if isinstance(tailCount, int) and tailCount < 100:
                lines += [
                    "",
                    f"Treat the `{last}` row with care: it is {tailCount} sentences, so "
                    f"its BLEU is noisy and the slope above rests largely on it.",
                ]

    lines += ["", "## Figures", "",
              "![BLEU by source length, both arms](bottleneck_by_length.png)", "",
              "![Arm comparison](arm_comparison.png)", ""]
    (outDir / "endterm_results.md").write_text("\n".join(lines) + "\n")
    print(f"[done] -> {outDir}")


if __name__ == "__main__":
    main()
