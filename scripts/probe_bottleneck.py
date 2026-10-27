#!/usr/bin/env python3
"""Measure the bottleneck directly, without going through BLEU or a beam.

Owner: M4 (analysis & writing). Promised in the mid-term report, section 9.

BLEU is a noisy, beam-mediated view of a model. Two sharper probes:

1. PER-LENGTH PERPLEXITY. Token perplexity on the test set, bucketed by source
   length, for each arm. This is the model's own likelihood -- no beam, no
   brevity penalty, no n-gram matching -- so if the fixed-vector arm's
   perplexity climbs with source length while the attention arm's stays flat,
   that is the bottleneck with the decoding machinery taken out of the picture.

2. SENTENCE-VECTOR SATURATION. The paper's model compresses every source
   sentence into one vector of fixed width. If that vector has to carry strictly
   more information as sentences grow, its norm cannot keep growing with it --
   so the per-length mean of ||v|| flattening is evidence of a capacity limit
   that does not depend on the decoder at all.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seq2seq.data import ParallelDataset, collate, make_batches  # noqa: E402
from seq2seq.evaluate import BUCKET_SETS, load_models  # noqa: E402
from seq2seq.plots import plot_bottleneck_probe  # noqa: E402
from seq2seq.train import load_prepared  # noqa: E402
from seq2seq.utils import pick_device  # noqa: E402


@torch.no_grad()
def perplexity_by_length(model, dataset, src_vocab, tgt_vocab, reverse, device,
                         buckets, batch_size=32):
    """Per-token perplexity within each source-length bucket."""
    model.eval()
    crit = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id, reduction="sum")
    out = []
    for lo, hi in buckets:
        idx = [i for i in range(len(dataset)) if lo <= len(dataset.src_ids[i]) <= hi]
        if len(idx) < 5:
            continue
        sub = ParallelDataset([dataset.src_ids[i] for i in idx],
                              [dataset.tgt_ids[i] for i in idx])
        nll = tokens = 0.0
        for batch in make_batches(sub, batch_size, shuffle=False):
            src, src_len, tgt_in, tgt_out = collate(
                sub, batch, src_vocab, tgt_vocab, reverse, device
            )
            logits = model(src, src_len, tgt_in)
            nll += crit(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)).item()
            tokens += int((tgt_out != tgt_vocab.pad_id).sum())
        out.append({"bucket": f"{lo}-{'inf' if hi > 10**5 else hi}", "n": len(idx),
                    "ppl": round(math.exp(nll / max(tokens, 1)), 3)})
    return out


@torch.no_grad()
def sentence_vector_norm_by_length(model, dataset, src_vocab, tgt_vocab, reverse,
                                   device, buckets, batch_size=32):
    """Mean ||v||_2 of the encoder's final (h, c), per source-length bucket."""
    model.eval()
    out = []
    for lo, hi in buckets:
        idx = [i for i in range(len(dataset)) if lo <= len(dataset.src_ids[i]) <= hi]
        if len(idx) < 5:
            continue
        sub = ParallelDataset([dataset.src_ids[i] for i in idx],
                              [dataset.tgt_ids[i] for i in idx])
        total, count = 0.0, 0
        for batch in make_batches(sub, batch_size, shuffle=False):
            src, src_len, _, _ = collate(sub, batch, src_vocab, tgt_vocab, reverse, device)
            h, c = model.encode(src, src_len)
            # One vector per sentence: all layers of h and c concatenated, which
            # is exactly the paper's "8000 real numbers" at their scale.
            v = torch.cat([h.permute(1, 0, 2).flatten(1), c.permute(1, 0, 2).flatten(1)], dim=1)
            total += v.norm(dim=1).sum().item()
            count += v.size(0)
        out.append({"bucket": f"{lo}-{'inf' if hi > 10**5 else hi}", "n": len(idx),
                    "mean_norm": round(total / count, 4)})
    return out


def main():
    ap = argparse.ArgumentParser(description="Direct probes of the fixed-vector bottleneck.")
    ap.add_argument("--base-checkpoint", required=True, help="the no-attention arm")
    ap.add_argument("--attn-checkpoint", default=None, help="the attention arm (optional)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="results/probe")
    ap.add_argument("--reverse-source", dest="reverse_source", action="store_true", default=False)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--buckets", default="endterm", choices=sorted(BUCKET_SETS))
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    args = ap.parse_args()

    device = pick_device(args.device)
    buckets = BUCKET_SETS[args.buckets]
    src_vocab, tgt_vocab, splits = load_prepared(Path(args.data_dir).expanduser())
    data = splits[args.split]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report = {"split": args.split, "reverse_source": args.reverse_source, "arms": {}}

    arms = [("base", args.base_checkpoint)]
    if args.attn_checkpoint:
        arms.append(("attn", args.attn_checkpoint))

    for arm, ckpt in arms:
        model = load_models([ckpt], src_vocab, tgt_vocab, device)[0][0]
        print(f"  [{arm}] attention={model.attention_kind or 'none'}")
        entry = {"checkpoint": ckpt, "attention": model.attention_kind or "none",
                 "ppl_by_length": perplexity_by_length(
                     model, data, src_vocab, tgt_vocab, args.reverse_source, device, buckets)}
        if arm == "base":
            entry["sentence_vector_norm"] = sentence_vector_norm_by_length(
                model, data, src_vocab, tgt_vocab, args.reverse_source, device, buckets)
            entry["sentence_vector_size"] = model.sentence_vector_size
            norms = entry["sentence_vector_norm"]
            plot_bottleneck_probe(
                [b["bucket"] for b in norms], [b["mean_norm"] for b in norms],
                out / "sentence_vector_norm.png",
                "Sentence-vector norm vs source length (fixed-vector arm)",
            )
        report["arms"][arm] = entry
        for row in entry["ppl_by_length"]:
            print(f"      {row['bucket']:>8s}  n={row['n']:>4d}  ppl={row['ppl']:.3f}")

    (out / "probe.json").write_text(json.dumps(report, indent=2))

    md = ["# Direct probes of the fixed-vector bottleneck", "",
          f"Split: `{args.split}`, "
          f"{'reversed' if args.reverse_source else 'forward'} source. No beam search is",
          "involved in any number on this page -- these are the models' own likelihoods.",
          "", "## Per-token perplexity by source length", ""]
    base = report["arms"]["base"]["ppl_by_length"]
    has_attn = "attn" in report["arms"]
    if has_attn:
        attn = {r["bucket"]: r["ppl"] for r in report["arms"]["attn"]["ppl_by_length"]}
        md += ["| source length | n | fixed vector | + attention | ratio |", "|---|---|---|---|---|"]
        for r in base:
            if r["bucket"] in attn:
                md.append(f"| {r['bucket']} | {r['n']} | {r['ppl']:.3f} | "
                          f"{attn[r['bucket']]:.3f} | {attn[r['bucket']] / r['ppl']:.3f} |")
        md += ["", "A ratio that falls as the buckets get longer means attention helps *more*",
               "on long sentences -- which is the bottleneck claim, stated in likelihood",
               "rather than in BLEU."]
    else:
        md += ["| source length | n | perplexity |", "|---|---|---|"]
        for r in base:
            md.append(f"| {r['bucket']} | {r['n']} | {r['ppl']:.3f} |")

    norms = report["arms"]["base"].get("sentence_vector_norm", [])
    if norms:
        size = report["arms"]["base"]["sentence_vector_size"]
        md += ["", "## Sentence-vector saturation", "",
               f"The fixed-length vector is {size:,} real numbers "
               "(num_layers x hidden x 2, the analogue of the paper's 8,000), and it carries",
               "the entire source sentence regardless of how long that sentence is.", "",
               "| source length | n | mean &#124;&#124;v&#124;&#124; |", "|---|---|---|"]
        for r in norms:
            md.append(f"| {r['bucket']} | {r['n']} | {r['mean_norm']:.4f} |")
        md += ["", "![Sentence-vector norm](sentence_vector_norm.png)"]
    (out / "probe.md").write_text("\n".join(md) + "\n")
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
