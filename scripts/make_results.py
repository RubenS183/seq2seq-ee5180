#!/usr/bin/env python3
"""Decode the whole grid, score it, and emit the reproduction table and figures.

Produces, under results/<name>/:
  table1_reproduction.csv   one row per (direction, beam, model set)
  results.md                the same table rendered, next to the paper's numbers
  bleu_by_length.png        Fig. 3 analogue
  beam_sweep.png            the Table 1 beam trend
  training_curves.png       dev perplexity, forward against reversed
  sample_translations.md    Table 3 analogue
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seq2seq.data import read_lines  # noqa: E402
from seq2seq.evaluate import BUCKET_SETS, run_eval  # noqa: E402
from seq2seq.plots import plot_beam_sweep, plot_bleu_by_length, plot_training_curves  # noqa: E402

PAPER = {
    ("fwd", 12, 1): 26.17,
    ("rev", 12, 1): 30.59,
    ("rev", 1, 5): 33.00,
    ("rev", 12, 2): 33.27,
    ("rev", 2, 5): 34.50,
    ("rev", 12, 5): 34.81,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--beams", type=int, nargs="+", default=[1, 2, 12])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    # Which end-term arm's run directories to read. "" is the mid-term layout,
    # {direction}_seed{N}, and "attn" reads {direction}_attn_seed{N}.
    parser.add_argument("--arm", default="", choices=["", "attn"])
    parser.add_argument("--buckets", default="midterm", choices=sorted(BUCKET_SETS))
    parser.add_argument("--max-decode-len", dest="max_decode_len", type=int, default=100)
    parser.add_argument("--directions", nargs="+", default=["fwd", "rev"],
                        choices=["fwd", "rev"])
    parser.add_argument("--ensemble", action="store_true",
                        help="also decode a seed-ensemble")
    parser.add_argument("--per-seed", dest="per_seed", action="store_true",
                        help="also decode each seed on its own at the largest beam, so "
                             "the end-term can report a mean and a spread rather than "
                             "one run's number")
    parser.add_argument("--limit", type=int, default=0)
    # See evaluate.py: CPU is 9x faster than MPS for beam search.
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scale-note", default="")
    parser.add_argument("--rebuild-md", action="store_true",
                        help="rewrite results.md and the figures from all_results.json, "
                             "without decoding again (decoding the full test set "
                             "takes ~30 min)")
    args = parser.parse_args()

    dataDir = Path(args.data_dir).expanduser()
    runsRoot = Path(args.runs_root).expanduser()
    outDir = Path(args.out).expanduser()
    outDir.mkdir(parents=True, exist_ok=True)
    hypDir = outDir / "hypotheses"

    results = []
    cachedPath = outDir / "all_results.json"
    if args.rebuild_md:
        if not cachedPath.exists():
            sys.exit(f"--rebuild-md needs {cachedPath}; run without it first.")
        results = json.loads(cachedPath.read_text())
        print(f"[rebuild-md] {len(results)} cached rows, no decoding")

    arm = f"_{args.arm}" if args.arm else ""
    armLabel = args.arm or "base"
    buckets = BUCKET_SETS[args.buckets]
    wanted = [(d, d == "rev") for d in ("fwd", "rev") if d in args.directions]

    for direction, reverse in ([] if args.rebuild_md else wanted):
        checkpoints = [runsRoot / f"{direction}{arm}_seed{s}" / "best.pt"
                       for s in args.seeds]
        present = [c for c in checkpoints if c.exists()]
        if not present:
            print(f"[warn] no checkpoints for {direction}: {checkpoints}", file=sys.stderr)
            continue

        for beam in args.beams:
            # Single model, meaning the primary seed.
            result = run_eval([present[0]], dataDir, beam, reverse, args.device,
                              args.limit, 0.0, "fr", "en", hypDir,
                              f"{direction}{arm}_seed{args.seeds[0]}",
                              buckets, args.max_decode_len)
            result["direction"], result["seeds"] = direction, [args.seeds[0]]
            result["arm"] = armLabel
            results.append(result)
            print(f"  {direction} beam={beam:>2} single -> "
                  f"BLEU_tok {result['bleu_tok']:.2f} "
                  f"| BLEU_detok {result['bleu_detok']:.2f} "
                  f"| ppl {result['test_ppl']}", flush=True)

        if args.per_seed and len(present) > 1:
            beam = max(args.beams)
            for checkpoint, seed in zip(present, args.seeds):
                # present[0] at this beam is already in `results` as the single
                # model row above. Decoding it again would both waste a decode
                # and leave two identical rows, which any mean over seeds then
                # weights double. See dedupe() in make_endterm_results.py, which
                # also guards already-written all_results.json files.
                if checkpoint == present[0] and beam in args.beams:
                    continue
                result = run_eval([checkpoint], dataDir, beam, reverse, args.device,
                                  args.limit, 0.0, "fr", "en", hypDir,
                                  f"{direction}{arm}_seed{seed}",
                                  buckets, args.max_decode_len)
                result["direction"], result["seeds"] = direction, [seed]
                result["arm"] = armLabel
                results.append(result)
                print(f"  {direction} beam={beam:>2} seed{seed} -> "
                      f"BLEU_tok {result['bleu_tok']:.2f}", flush=True)

        if args.ensemble and len(present) > 1:
            for beam in args.beams:
                result = run_eval(present, dataDir, beam, reverse, args.device,
                                  args.limit, 0.0, "fr", "en", hypDir,
                                  f"{direction}{arm}_ens{len(present)}",
                                  buckets, args.max_decode_len)
                result["direction"] = direction
                result["seeds"] = list(args.seeds[: len(present)])
                result["arm"] = armLabel
                results.append(result)
                print(f"  {direction} beam={beam:>2} ens{len(present)} -> "
                      f"BLEU_tok {result['bleu_tok']:.2f}", flush=True)

    if not args.rebuild_md:
        cachedPath.write_text(json.dumps(results, indent=2))

    with (outDir / "table1_reproduction.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["direction", "n_models", "seeds", "beam", "bleu_tok",
                         "bleu_detok", "test_ppl", "n_sentences", "paper_bleu"])
        for row in results:
            writer.writerow([
                row["direction"], row["n_models"], "+".join(map(str, row["seeds"])),
                row["beam_size"], row["bleu_tok"], row["bleu_detok"], row["test_ppl"],
                row["n_sentences"],
                PAPER.get((row["direction"], row["beam_size"], row["n_models"]), ""),
            ])

    plot_bleu_by_length(
        [r for r in results if r["beam_size"] == max(args.beams) and r["n_models"] == 1],
        outDir / "bleu_by_length.png",
        f"BLEU by source length ({args.name}) - paper Fig. 3 analogue",
    )
    plot_beam_sweep(results, outDir / "beam_sweep.png", f"Beam sweep ({args.name})")
    plot_training_curves(
        # ("rev", "fwd") order, not args.directions order: it sets the legend and
        # draw order, and flipping it changes the rendered PNG for runs that are
        # otherwise identical.
        {f"{d}{arm}_seed{s}": runsRoot / f"{d}{arm}_seed{s}"
         for d in ("rev", "fwd") if d in args.directions
         for s in args.seeds
         if (runsRoot / f"{d}{arm}_seed{s}" / "log.jsonl").exists()},
        outDir / "training_curves.png", f"Dev perplexity ({args.name})",
    )

    lines = [f"# Reproduction results - {args.name}", ""]
    if args.scale_note:
        lines += [f"> **Scale gap.** {args.scale_note}", ""]
    lines += ["| Method | beam | our BLEU (tok, cased) | our BLEU (sacreBLEU detok) "
              "| our test ppl | paper BLEU |",
              "|---|---|---|---|---|---|"]
    ordered = sorted(results, key=lambda r: (r["direction"] != "fwd", r["n_models"],
                                             r["beam_size"]))
    for row in ordered:
        label = ("Single " if row["n_models"] == 1
                 else f"Ensemble of {row['n_models']} ") + \
                ("forward" if row["direction"] == "fwd" else "reversed") + " LSTM"
        paperBleu = PAPER.get((row["direction"], row["beam_size"], row["n_models"]), "")
        lines.append(f"| {label} | {row['beam_size']} | **{row['bleu_tok']:.2f}** | "
                     f"{row['bleu_detok']:.2f} | "
                     f"{row['test_ppl'] if row['test_ppl'] else '-'} | "
                     f"{paperBleu if paperBleu else '-'} |")

    singles = {(r["direction"], r["beam_size"]): r for r in results if r["n_models"] == 1}
    if ("fwd", 12) in singles and ("rev", 12) in singles:
        forward12, reversed12 = singles[("fwd", 12)], singles[("rev", 12)]
        gain = reversed12["bleu_tok"] - forward12["bleu_tok"]
        lines += ["", "## The paper's central claim, at our scale", "",
                  f"- BLEU (beam 12): forward **{forward12['bleu_tok']:.2f}** -> "
                  f"reversed **{reversed12['bleu_tok']:.2f}** (**{gain:+.2f}**); "
                  f"paper 26.17 -> 30.59 (+4.42).",
                  f"- Test perplexity: forward **{forward12['test_ppl']}** -> "
                  f"reversed **{reversed12['test_ppl']}**; paper 5.8 -> 4.7."]

    if all(("rev", b) in singles for b in (1, 2, 12)):
        bleu1, bleu2, bleu12 = (singles[("rev", b)]["bleu_tok"] for b in (1, 2, 12))
        head = f"- Beam: B=1 {bleu1:.2f}, B=2 {bleu2:.2f}, B=12 {bleu12:.2f} - "
        if bleu12 > bleu2 > bleu1:
            # The paper's ordering, so report how much of the gain B=2 captures.
            recovered = (bleu2 - bleu1) / (bleu12 - bleu1) * 100
            lines.append(head + f"B=2 recovers {recovered:.0f}% of the B=1 -> B=12 "
                                "gain, as in the paper (sec. 3.2).")
        elif bleu12 < bleu2:
            # Non-monotone, so a percentage "recovered" would be meaningless and
            # can exceed 100%. Describe what actually happened instead.
            lines.append(head + f"BLEU peaks at B=2 and falls by {bleu2 - bleu12:.2f} "
                                "at B=12, so the paper's monotone beam trend does NOT "
                                "reproduce at our scale. A wider beam does find "
                                "higher-probability hypotheses, but they are worse "
                                "translations. Length normalisation removes the "
                                "brevity penalty and makes BLEU worse still, so "
                                "brevity is a symptom rather than the cause - see "
                                "beam_ablation.md for the decomposition.")
        else:
            lines.append(head + "the ordering is non-monotone; see the per-beam figures.")

    lines += ["", "## Figures", "",
              "![BLEU by source length](bleu_by_length.png)", "",
              "![Beam sweep](beam_sweep.png)", "",
              "![Training curves](training_curves.png)", "",
              "## Scoring signatures", ""]
    if results:
        lines += [f"- tokenized (multi-bleu.pl equivalent): `{results[0]['sig_tok']}`",
                  f"- detokenized sacreBLEU: `{results[0]['sig_detok']}`"]
    (outDir / "results.md").write_text("\n".join(lines) + "\n")

    best = max((r for r in results if r["direction"] == "rev"),
               key=lambda r: r["bleu_tok"], default=None)
    if best:
        stem = (f"rev{arm}_ens{best['n_models']}" if best["n_models"] > 1
                else f"rev{arm}_seed{best['seeds'][0]}") + f"_beam{best['beam_size']}"
        hypPath = hypDir / f"{stem}.hyp.detok.fr"
        if hypPath.exists():
            sources = read_lines(dataDir / "test.raw.en")
            references = read_lines(dataDir / "test.raw.fr")
            hypotheses = read_lines(hypPath)
            shown = min(len(hypotheses), 400)
            markdown = [
                f"# Sample translations - {args.name} (paper Table 3 analogue)", "",
                f"Model: reversed, {best['n_models']} model(s), "
                f"beam {best['beam_size']}.", "",
            ]
            for i in range(0, shown, max(1, shown // 12)):
                markdown += [f"**Source**  {sources[i]}", "",
                             f"**Ours**    {hypotheses[i]}", "",
                             f"**Reference**  {references[i]}", "", "---", ""]
            (outDir / "sample_translations.md").write_text("\n".join(markdown))

    print(f"[done] -> {outDir}")


if __name__ == "__main__":
    main()
