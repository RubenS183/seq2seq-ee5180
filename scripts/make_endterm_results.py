#!/usr/bin/env python3
"""Join the two end-term arms into the bottleneck result.

Owner: M4 (analysis & writing).

`make_results.py` produces one arm's table at a time. The end-term's claim is
about the DIFFERENCE between arms, so this script reads both and emits:

  bottleneck_by_length.png   both arms' BLEU-vs-source-length on one axis
  arm_comparison.png         the 2x2 grid as bars, with a seed spread if present
  endterm_results.md         the comparison table and the claims, stated with
                             their scale caveat
  endterm_results.json       every row that went into the above

The headline is deliberately NOT "attention scores higher". It is the SHAPE of
the length curve: a fixed-length vector has to carry a whole sentence, so it
should degrade as sentences grow, and attention should not. A model that simply
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
    for r in rows:
        r.setdefault("arm", arm)
    return rows


def singles(rows, beam=None):
    out = [r for r in rows if r["n_models"] == 1]
    return [r for r in out if beam is None or r["beam_size"] == beam]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", required=True, help="results dir of the no-attention arm")
    ap.add_argument("--attn", required=True, help="results dir of the attention arm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beam", type=int, default=12, help="beam for the headline table")
    ap.add_argument("--scale-note", default="")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = (load_rows(Path(args.base) / "all_results.json", "base")
            + load_rows(Path(args.attn) / "all_results.json", "attn"))
    if not rows:
        sys.exit("no result rows found in either arm; run make_results.py first")
    (out / "endterm_results.json").write_text(json.dumps(rows, indent=2))

    # ---- the headline figure: both arms' length curves on one axis ----------
    head = [r for r in singles(rows, args.beam) if len(r["seeds"]) == 1]
    # one curve per (direction, arm); if several seeds exist take the first
    seen, curves = set(), []
    for r in head:
        key = (r["direction"], r["arm"])
        if key not in seen:
            seen.add(key)
            curves.append(r)
    plot_bleu_by_length_overlay(
        curves, out / "bottleneck_by_length.png",
        f"BLEU by source length, fixed vector vs attention ({args.name})",
    )

    # ---- the 2x2 grid as bars, with a seed spread where we have one ---------
    bars = []
    for direction in ("fwd", "rev"):
        for arm in ("base", "attn"):
            scores = [r["bleu_tok"] for r in singles(rows, args.beam)
                      if r["direction"] == direction and r["arm"] == arm
                      and len(r["seeds"]) == 1]
            if not scores:
                continue
            bars.append({
                "label": f"{'reversed' if direction == 'rev' else 'forward'}\n{ARM_LABEL[arm]}",
                "mean": statistics.fmean(scores),
                "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "n": len(scores),
                "reverse": direction == "rev",
                "arm": arm,
                "direction": direction,
            })
    if bars:
        plot_arm_bars(bars, out / "arm_comparison.png",
                      f"BLEU at beam {args.beam} ({args.name})")

    # ---- the table ---------------------------------------------------------
    lines = [f"# End-term results - {args.name}", ""]
    if args.scale_note:
        lines += [f"> **Scale gap.** {args.scale_note}", ""]
    lines += [
        "The two arms share their corpus, vocabulary, seed, batch order, optimiser,",
        "learning-rate schedule and gradient clipping. `attention: bahdanau` is the",
        "only difference between them.", "",
        "| direction | arm | beam | BLEU (tok, cased) | BLEU (detok) | test ppl | params | runs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for b in bars:
        matching = [r for r in singles(rows, args.beam)
                    if r["direction"] == b["direction"] and r["arm"] == b["arm"]
                    and len(r["seeds"]) == 1]
        detok = statistics.fmean([r["bleu_detok"] for r in matching])
        ppls = [r["test_ppl"] for r in matching if r.get("test_ppl")]
        params = matching[0].get("n_params")
        spread = f" ± {b['std']:.2f}" if b["n"] > 1 else ""
        lines.append(
            f"| {'reversed' if b['reverse'] else 'forward'} | {ARM_LABEL[b['arm']]} | "
            f"{args.beam} | **{b['mean']:.2f}**{spread} | {detok:.2f} | "
            f"{statistics.fmean(ppls):.3f} | {params/1e6:.1f}M | {b['n']} |"
            if ppls else
            f"| {'reversed' if b['reverse'] else 'forward'} | {ARM_LABEL[b['arm']]} | "
            f"{args.beam} | **{b['mean']:.2f}**{spread} | {detok:.2f} | - | "
            f"{params/1e6:.1f}M | {b['n']} |"
        )

    by = {(b["direction"], b["arm"]): b for b in bars}
    lines += ["", "## What the arms say about the bottleneck", ""]

    for direction in ("rev", "fwd"):
        if (direction, "base") in by and (direction, "attn") in by:
            lo, hi = by[(direction, "base")]["mean"], by[(direction, "attn")]["mean"]
            name = "reversed" if direction == "rev" else "forward"
            lines.append(f"- {name}: fixed vector **{lo:.2f}** -> attention **{hi:.2f}** "
                         f"(**{hi - lo:+.2f}** BLEU).")

    # Does attention make the reversal trick unnecessary? The paper reverses the
    # source to shorten the lag between a source word and the step that needs
    # it. If the bottleneck is what reversal was working around, the gap should
    # shrink once attention exists.
    if all(k in by for k in (("fwd", "base"), ("rev", "base"), ("fwd", "attn"), ("rev", "attn"))):
        gap_base = by[("rev", "base")]["mean"] - by[("fwd", "base")]["mean"]
        gap_attn = by[("rev", "attn")]["mean"] - by[("fwd", "attn")]["mean"]
        lines += [
            "",
            f"- The reversal gap (reversed minus forward) is **{gap_base:+.2f}** without "
            f"attention and **{gap_attn:+.2f}** with it.",
        ]
        if abs(gap_attn) < abs(gap_base):
            shrink = (1 - abs(gap_attn) / abs(gap_base)) * 100 if gap_base else 0.0
            lines.append(
                f"  Attention absorbs {shrink:.0f}% of it. That is the paper's trick being "
                "made redundant: reversing the source shortens the distance between a "
                "source word and the decoder step that needs it, and attention removes "
                "the distance altogether."
            )
        else:
            lines.append(
                "  The gap did **not** shrink at this scale. Reported rather than hidden: "
                "it is evidence against the strong reading of the bottleneck story, and "
                "the length curves below are the better-powered test."
            )

    # The shape claim, bucket by bucket.
    lines += ["", "### The shape of the length curve", "",
              "This is the actual bottleneck claim: a single fixed vector should lose ground",
              "as the source grows, and attention should not.", ""]
    rev = {r["arm"]: r for r in curves if r["direction"] == "rev"}
    if "base" in rev and "attn" in rev:
        bb = {b["bucket"]: b["bleu_tok"] for b in rev["base"].get("bleu_by_length", [])}
        ba = {b["bucket"]: b["bleu_tok"] for b in rev["attn"].get("bleu_by_length", [])}
        lines += ["| source length | fixed vector | + attention | gain |", "|---|---|---|---|"]
        for bucket in [b["bucket"] for b in rev["base"].get("bleu_by_length", [])]:
            if bucket in ba:
                lines.append(f"| {bucket} | {bb[bucket]:.2f} | {ba[bucket]:.2f} | "
                             f"**{ba[bucket] - bb[bucket]:+.2f}** |")
        buckets = [b for b in bb if b in ba]
        if len(buckets) >= 2:
            first, last = buckets[0], buckets[-1]
            lines += [
                "",
                f"- Fixed vector, {first} -> {last}: {bb[first]:.2f} -> {bb[last]:.2f} "
                f"({bb[last] - bb[first]:+.2f}).",
                f"- Attention,    {first} -> {last}: {ba[first]:.2f} -> {ba[last]:.2f} "
                f"({ba[last] - ba[first]:+.2f}).",
                "",
                "If the first of those falls and the second does not, the fixed-length vector",
                "was the binding constraint. If both fall equally, something else is.",
            ]

    lines += ["", "## Figures", "",
              "![BLEU by source length, both arms](bottleneck_by_length.png)", "",
              "![Arm comparison](arm_comparison.png)", ""]
    (out / "endterm_results.md").write_text("\n".join(lines) + "\n")
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
