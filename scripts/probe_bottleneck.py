#!/usr/bin/env python3
"""Measure the bottleneck directly, without going through BLEU or a beam.

BLEU is a noisy, beam-mediated view of a model. Two sharper probes:

1. PER-LENGTH PERPLEXITY. Token perplexity on the test set, bucketed by source
   length, for each arm. This is the model's own likelihood, with no beam, no
   brevity penalty and no n-gram matching, so if the fixed-vector arm's
   perplexity climbs with source length while the attention arm's stays flat,
   that is the bottleneck with the decoding machinery taken out of the picture.

2. SENTENCE-VECTOR SATURATION. The paper's model compresses every source sentence
   into one vector of fixed width. If that vector has to carry strictly more
   information as sentences grow, its norm cannot keep growing with it, so the
   per-length mean of ||v|| flattening is evidence of a capacity limit that does
   not depend on the decoder at all.
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


def _bucket_label(low, high):
    return f"{low}-{'inf' if high > 10**5 else high}"


def _bucket_subset(dataset, low, high):
    """The dataset restricted to sentences whose source length is in the bucket."""
    selected = [
        i for i in range(len(dataset)) if low <= len(dataset.src_ids[i]) <= high
    ]
    if len(selected) < 5:
        return None, selected
    subset = ParallelDataset([dataset.src_ids[i] for i in selected],
                             [dataset.tgt_ids[i] for i in selected])
    return subset, selected


@torch.no_grad()
def perplexity_by_length(model, dataset, srcVocab, tgtVocab, reverse, device,
                         buckets, batchSize=32):
    """Per-token perplexity within each source-length bucket."""
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=tgtVocab.pad_id, reduction="sum")
    rows = []
    for low, high in buckets:
        subset, selected = _bucket_subset(dataset, low, high)
        if subset is None:
            continue
        totalNll = totalTokens = 0.0
        for batch in make_batches(subset, batchSize, shuffle=False):
            src, srcLen, tgtIn, tgtOut = collate(
                subset, batch, srcVocab, tgtVocab, reverse, device
            )
            logits = model(src, srcLen, tgtIn)
            totalNll += criterion(
                logits.reshape(-1, logits.size(-1)), tgtOut.reshape(-1)
            ).item()
            totalTokens += int((tgtOut != tgtVocab.pad_id).sum())
        rows.append({"bucket": _bucket_label(low, high), "n": len(selected),
                     "ppl": round(math.exp(totalNll / max(totalTokens, 1)), 3)})
    return rows


@torch.no_grad()
def sentence_vector_norm_by_length(model, dataset, srcVocab, tgtVocab, reverse,
                                   device, buckets, batchSize=32):
    """Mean ||v||_2 of the encoder's final (h, c), per source-length bucket."""
    model.eval()
    rows = []
    for low, high in buckets:
        subset, selected = _bucket_subset(dataset, low, high)
        if subset is None:
            continue
        normSum, sentences = 0.0, 0
        for batch in make_batches(subset, batchSize, shuffle=False):
            src, srcLen, _, _ = collate(
                subset, batch, srcVocab, tgtVocab, reverse, device
            )
            hidden, cell = model.encode(src, srcLen)
            # One vector per sentence: all layers of h and c concatenated, which
            # is exactly the paper's "8000 real numbers" at their scale.
            sentenceVector = torch.cat(
                [hidden.permute(1, 0, 2).flatten(1), cell.permute(1, 0, 2).flatten(1)],
                dim=1,
            )
            normSum += sentenceVector.norm(dim=1).sum().item()
            sentences += sentenceVector.size(0)
        rows.append({"bucket": _bucket_label(low, high), "n": len(selected),
                     "mean_norm": round(normSum / sentences, 4)})
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Direct probes of the fixed-vector bottleneck."
    )
    parser.add_argument("--base-checkpoint", required=True, help="the no-attention arm")
    parser.add_argument("--attn-checkpoint", default=None,
                        help="the attention arm (optional)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default="results/probe")
    parser.add_argument("--reverse-source", dest="reverse_source",
                        action="store_true", default=False)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--buckets", default="endterm", choices=sorted(BUCKET_SETS))
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    args = parser.parse_args()

    device = pick_device(args.device)
    buckets = BUCKET_SETS[args.buckets]
    srcVocab, tgtVocab, splits = load_prepared(Path(args.data_dir).expanduser())
    dataset = splits[args.split]
    outDir = Path(args.out)
    outDir.mkdir(parents=True, exist_ok=True)

    report = {"split": args.split, "reverse_source": args.reverse_source, "arms": {}}

    arms = [("base", args.base_checkpoint)]
    if args.attn_checkpoint:
        arms.append(("attn", args.attn_checkpoint))

    for arm, checkpointPath in arms:
        model = load_models([checkpointPath], srcVocab, tgtVocab, device)[0][0]
        print(f"  [{arm}] attention={model.attention_kind or 'none'}")
        entry = {
            "checkpoint": checkpointPath,
            "attention": model.attention_kind or "none",
            "ppl_by_length": perplexity_by_length(
                model, dataset, srcVocab, tgtVocab, args.reverse_source,
                device, buckets,
            ),
        }
        if arm == "base":
            entry["sentence_vector_norm"] = sentence_vector_norm_by_length(
                model, dataset, srcVocab, tgtVocab, args.reverse_source,
                device, buckets,
            )
            entry["sentence_vector_size"] = model.sentence_vector_size
            norms = entry["sentence_vector_norm"]
            plot_bottleneck_probe(
                [row["bucket"] for row in norms], [row["mean_norm"] for row in norms],
                outDir / "sentence_vector_norm.png",
                "Sentence-vector norm vs source length (fixed-vector arm)",
            )
        report["arms"][arm] = entry
        for row in entry["ppl_by_length"]:
            print(f"      {row['bucket']:>8s}  n={row['n']:>4d}  ppl={row['ppl']:.3f}")

    (outDir / "probe.json").write_text(json.dumps(report, indent=2))

    markdown = ["# Direct probes of the fixed-vector bottleneck", "",
                f"Split: `{args.split}`, "
                f"{'reversed' if args.reverse_source else 'forward'} source. "
                "No beam search is",
                "involved in any number on this page, these are the models' own "
                "likelihoods.",
                "", "## Per-token perplexity by source length", ""]
    baseRows = report["arms"]["base"]["ppl_by_length"]
    if "attn" in report["arms"]:
        attnPpl = {r["bucket"]: r["ppl"] for r in report["arms"]["attn"]["ppl_by_length"]}
        markdown += ["| source length | n | fixed vector | + attention | ratio |",
                     "|---|---|---|---|---|"]
        for row in baseRows:
            if row["bucket"] in attnPpl:
                markdown.append(
                    f"| {row['bucket']} | {row['n']} | {row['ppl']:.3f} | "
                    f"{attnPpl[row['bucket']]:.3f} | "
                    f"{attnPpl[row['bucket']] / row['ppl']:.3f} |"
                )
        markdown += ["",
                     "A ratio that falls as the buckets get longer means attention "
                     "helps *more*",
                     "on long sentences, which is the bottleneck claim stated in "
                     "likelihood rather than in BLEU."]
    else:
        markdown += ["| source length | n | perplexity |", "|---|---|---|"]
        for row in baseRows:
            markdown.append(f"| {row['bucket']} | {row['n']} | {row['ppl']:.3f} |")

    norms = report["arms"]["base"].get("sentence_vector_norm", [])
    if norms:
        vectorSize = report["arms"]["base"]["sentence_vector_size"]
        markdown += ["", "## Sentence-vector saturation", "",
                     f"The fixed-length vector is {vectorSize:,} real numbers "
                     "(num_layers x hidden x 2, the analogue of the paper's 8,000), "
                     "and it carries",
                     "the entire source sentence regardless of how long that "
                     "sentence is.", "",
                     "| source length | n | mean &#124;&#124;v&#124;&#124; |",
                     "|---|---|---|"]
        for row in norms:
            markdown.append(f"| {row['bucket']} | {row['n']} | {row['mean_norm']:.4f} |")
        markdown += ["", "![Sentence-vector norm](sentence_vector_norm.png)"]
    (outDir / "probe.md").write_text("\n".join(markdown) + "\n")
    print(f"[done] -> {outDir}")


if __name__ == "__main__":
    main()
