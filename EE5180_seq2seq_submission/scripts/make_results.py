#!/usr/bin/env python3
"""Decode the whole grid, score it, and emit the reproduction table + figures.

Owner: M3 (eval) / M4 (analysis).

Produces, under results/<name>/:
  table1_reproduction.csv   one row per (direction, beam, model set)
  results.md                the same table rendered, next to the paper's numbers
  bleu_by_length.png        Fig. 3 analogue
  beam_sweep.png            the Table 1 beam trend
  training_curves.png       dev perplexity, forward vs reversed
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
from seq2seq.evaluate import run_eval  # noqa: E402
from seq2seq.plots import plot_beam_sweep, plot_bleu_by_length, plot_training_curves  # noqa: E402
from seq2seq.utils import work_dir  # noqa: E402

PAPER = {
    ("fwd", 12, 1): 26.17,
    ("rev", 12, 1): 30.59,
    ("rev", 1, 5): 33.00,
    ("rev", 12, 2): 33.27,
    ("rev", 2, 5): 34.50,
    ("rev", 12, 5): 34.81,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beams", type=int, nargs="+", default=[1, 2, 12])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--ensemble", action="store_true", help="also decode a seed-ensemble")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")  # see evaluate.py: 9x faster than MPS for beam search
    ap.add_argument("--scale-note", default="")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    runs_root = Path(args.runs_root).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    hyp_dir = out / "hypotheses"

    results = []
    for direction, reverse in (("fwd", False), ("rev", True)):
        ckpts = [runs_root / f"{direction}_seed{s}" / "best.pt" for s in args.seeds]
        present = [c for c in ckpts if c.exists()]
        if not present:
            print(f"[warn] no checkpoints for {direction}: {ckpts}", file=sys.stderr)
            continue
        for beam in args.beams:
            # single model = the primary seed
            res = run_eval([present[0]], data_dir, beam, reverse, args.device,
                           args.limit, 0.0, "fr", "en", hyp_dir, f"{direction}_seed{args.seeds[0]}")
            res["direction"], res["seeds"] = direction, [args.seeds[0]]
            results.append(res)
            print(f"  {direction} beam={beam:>2} single -> BLEU_tok {res['bleu_tok']:.2f} "
                  f"| BLEU_detok {res['bleu_detok']:.2f} | ppl {res['test_ppl']}", flush=True)
        if args.ensemble and len(present) > 1:
            for beam in args.beams:
                res = run_eval(present, data_dir, beam, reverse, args.device,
                               args.limit, 0.0, "fr", "en", hyp_dir, f"{direction}_ens{len(present)}")
                res["direction"], res["seeds"] = direction, list(args.seeds[: len(present)])
                results.append(res)
                print(f"  {direction} beam={beam:>2} ens{len(present)} -> BLEU_tok {res['bleu_tok']:.2f}", flush=True)

    (out / "all_results.json").write_text(json.dumps(results, indent=2))

    # ---- CSV -------------------------------------------------------------
    with (out / "table1_reproduction.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["direction", "n_models", "seeds", "beam", "bleu_tok", "bleu_detok",
                    "test_ppl", "n_sentences", "paper_bleu"])
        for r in results:
            w.writerow([r["direction"], r["n_models"], "+".join(map(str, r["seeds"])),
                        r["beam_size"], r["bleu_tok"], r["bleu_detok"], r["test_ppl"],
                        r["n_sentences"], PAPER.get((r["direction"], r["beam_size"], r["n_models"]), "")])

    # ---- figures ---------------------------------------------------------
    plot_bleu_by_length([r for r in results if r["beam_size"] == max(args.beams) and r["n_models"] == 1],
                        out / "bleu_by_length.png",
                        f"BLEU by source length ({args.name}) - paper Fig. 3 analogue")
    plot_beam_sweep(results, out / "beam_sweep.png", f"Beam sweep ({args.name})")
    plot_training_curves(
        {f"{d}_seed{s}": runs_root / f"{d}_seed{s}" for d in ("rev", "fwd") for s in args.seeds
         if (runs_root / f"{d}_seed{s}" / "log.jsonl").exists()},
        out / "training_curves.png", f"Dev perplexity ({args.name})")

    # ---- markdown --------------------------------------------------------
    lines = [f"# Reproduction results - {args.name}", ""]
    if args.scale_note:
        lines += [f"> **Scale gap.** {args.scale_note}", ""]
    lines += ["| Method | beam | our BLEU (tok, cased) | our BLEU (sacreBLEU detok) | our test ppl | paper BLEU |",
              "|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: (r["direction"] != "fwd", r["n_models"], r["beam_size"])):
        name = ("Single " if r["n_models"] == 1 else f"Ensemble of {r['n_models']} ") + \
               ("forward" if r["direction"] == "fwd" else "reversed") + " LSTM"
        paper = PAPER.get((r["direction"], r["beam_size"], r["n_models"]), "")
        lines.append(f"| {name} | {r['beam_size']} | **{r['bleu_tok']:.2f}** | {r['bleu_detok']:.2f} | "
                     f"{r['test_ppl'] if r['test_ppl'] else '-'} | {paper if paper else '-'} |")
    single = {(r["direction"], r["beam_size"]): r for r in results if r["n_models"] == 1}
    if ("fwd", 12) in single and ("rev", 12) in single:
        f12, r12 = single[("fwd", 12)], single[("rev", 12)]
        lines += ["", "## The paper's central claim, at our scale", "",
                  f"- BLEU (beam 12): forward **{f12['bleu_tok']:.2f}** -> reversed **{r12['bleu_tok']:.2f}** "
                  f"(**{r12['bleu_tok'] - f12['bleu_tok']:+.2f}**); paper 26.17 -> 30.59 (+4.42).",
                  f"- Test perplexity: forward **{f12['test_ppl']}** -> reversed **{r12['test_ppl']}**; "
                  f"paper 5.8 -> 4.7."]
    if ("rev", 2) in single and ("rev", 12) in single and ("rev", 1) in single:
        b1, b2, b12 = (single[("rev", b)]["bleu_tok"] for b in (1, 2, 12))
        frac = (b2 - b1) / (b12 - b1) * 100 if abs(b12 - b1) > 1e-9 else float("nan")
        lines += [f"- Beam: B=1 {b1:.2f}, B=2 {b2:.2f}, B=12 {b12:.2f} - "
                  f"B=2 recovers {frac:.0f}% of the B=1 -> B=12 gain."]
    lines += ["", "## Figures", "",
              "![BLEU by source length](bleu_by_length.png)", "",
              "![Beam sweep](beam_sweep.png)", "",
              "![Training curves](training_curves.png)", "",
              "## Scoring signatures", ""]
    if results:
        lines += [f"- tokenized (multi-bleu.pl equivalent): `{results[0]['sig_tok']}`",
                  f"- detokenized sacreBLEU: `{results[0]['sig_detok']}`"]
    (out / "results.md").write_text("\n".join(lines) + "\n")

    # ---- sample translations (Table 3 analogue) --------------------------
    best = max((r for r in results if r["direction"] == "rev"), key=lambda r: r["bleu_tok"], default=None)
    if best:
        stem = f"{'rev_ens' + str(best['n_models']) if best['n_models'] > 1 else 'rev_seed' + str(best['seeds'][0])}_beam{best['beam_size']}"
        hyp_file = hyp_dir / f"{stem}.hyp.detok.fr"
        if hyp_file.exists():
            src = read_lines(data_dir / "test.raw.en")
            ref = read_lines(data_dir / "test.raw.fr")
            hyp = read_lines(hyp_file)
            md = [f"# Sample translations - {args.name} (paper Table 3 analogue)", "",
                  f"Model: reversed, {best['n_models']} model(s), beam {best['beam_size']}.", ""]
            for i in range(0, min(len(hyp), 400), max(1, min(len(hyp), 400) // 12)):
                md += [f"**Source**  {src[i]}", "", f"**Ours**    {hyp[i]}", "",
                       f"**Reference**  {ref[i]}", "", "---", ""]
            (out / "sample_translations.md").write_text("\n".join(md))

    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
