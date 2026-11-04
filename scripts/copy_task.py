#!/usr/bin/env python3
"""The bottleneck in miniature: a copy task both arms are asked to solve.

The end-term's claim about real translation takes ~11 GPU-hours to support. This
script makes the same point in about a minute, on a task with no confounds.

Target == source over uniformly random tokens. Nothing can be memorised and there
is no language model to lean on, so emitting position t REQUIRES reading source
position t. A decoder that can only see a fixed-length summary of the whole
sequence has to store every token in it, while attention just looks.

Measured result (seed 0, 1 layer x 64, 8 tokens): the attention arm reaches
near-zero loss and a clean diagonal alignment, while the fixed-vector arm is
still far from solving it. That is the bottleneck, isolated.

Adam, not the paper's SGD, because this is a controlled demonstration rather than
a reported reproduction and plain SGD at lr 0.7 needs far longer to leave chance
on a model this small. Stated here rather than buried.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seq2seq.attention import source_mask  # noqa: E402
from seq2seq.model import Seq2Seq  # noqa: E402
from seq2seq.plots import plot_alignment_heatmap  # noqa: E402
from seq2seq.vocab import Vocab  # noqa: E402


def alignment_spread(model, src, srcLen, tgtIn) -> float:
    """Max-minus-min weight over REAL source positions (padding sits at 0.0)."""
    weights = model.alignments(src, srcLen, tgtIn)
    if weights is None:
        return float("nan")
    padding = source_mask(srcLen, src.size(1)).unsqueeze(1).expand_as(weights)
    highest = weights.masked_fill(padding, float("-inf")).amax(-1)
    lowest = weights.masked_fill(padding, float("inf")).amin(-1)
    return (highest - lowest).max().item()


def run_arm(attention, vocab, length, batchSize, width, steps, lr, seed, logEvery):
    torch.manual_seed(seed)
    model = Seq2Seq(len(vocab), len(vocab), embDim=width, hiddenDim=width,
                    numLayers=1, srcPadId=vocab.pad_id, tgtPadId=vocab.pad_id,
                    attention=attention)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id, reduction="sum")
    generator = torch.Generator().manual_seed(seed)

    def makeBatch():
        src = torch.randint(4, len(vocab), (batchSize, length), generator=generator)
        return (torch.cat([src, torch.full((batchSize, 1), vocab.eos_id)], 1),
                torch.full((batchSize,), length + 1),
                torch.cat([torch.full((batchSize, 1), vocab.sos_id), src], 1),
                torch.cat([src, torch.full((batchSize, 1), vocab.eos_id)], 1))

    curve = []
    src = srcLen = tgtIn = None
    for step in range(1, steps + 1):
        src, srcLen, tgtIn, tgtOut = makeBatch()
        loss = criterion(model(src, srcLen, tgtIn).reshape(-1, len(vocab)),
                         tgtOut.reshape(-1)) / batchSize
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % logEvery == 0 or step == 1:
            curve.append({"step": step, "loss_per_seq": round(loss.item(), 4)})
            print(f"    step {step:5d}  loss/seq {loss.item():8.3f}", flush=True)
    return model, curve, (src, srcLen, tgtIn)


def main():
    parser = argparse.ArgumentParser(description="Copy-task bottleneck demonstration.")
    parser.add_argument("--out", default="results/copy_task")
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", dest="log_every", type=int, default=300)
    args = parser.parse_args()

    outDir = Path(args.out)
    outDir.mkdir(parents=True, exist_ok=True)
    vocab = Vocab(["<pad>", "<unk>", "<sos>", "<eos>"] + [f"w{i}" for i in range(20)])
    chanceLoss = (args.length + 1) * torch.tensor(float(len(vocab))).log().item()

    report = {
        "config": vars(args),
        "chance_loss_per_seq": round(chanceLoss, 3),
        "arms": {},
    }
    for name, attention in (("fixed vector", None), ("attention", "bahdanau")):
        print(f"  [{name}]", flush=True)
        model, curve, lastBatch = run_arm(
            attention, vocab, args.length, args.batch, args.width, args.steps,
            args.lr, args.seed, args.log_every,
        )
        entry = {"final_loss_per_seq": curve[-1]["loss_per_seq"], "curve": curve}
        if attention:
            src, srcLen, tgtIn = lastBatch
            entry["alignment_spread"] = round(
                alignment_spread(model, src, srcLen, tgtIn), 4
            )
            weights = model.alignments(src, srcLen, tgtIn)[0]
            peakPositions = weights[: args.length].argmax(-1).tolist()
            entry["argmax_source_position"] = peakPositions
            entry["is_diagonal"] = peakPositions == list(range(args.length))
            sourceLabels = [f"s{i}" for i in range(args.length)] + ["<eos>"]
            targetLabels = [f"t{i}" for i in range(args.length)] + ["<eos>"]
            plot_alignment_heatmap(
                weights.cpu().numpy(), sourceLabels, targetLabels,
                outDir / "copy_task_alignment.png",
                "Copy task: learned alignment (attention arm)",
            )
        report["arms"][name] = entry
        print(f"    -> final loss/seq {entry['final_loss_per_seq']}  "
              f"(chance {chanceLoss:.1f})\n", flush=True)

    (outDir / "copy_task.json").write_text(json.dumps(report, indent=2))

    fixedLoss = report["arms"]["fixed vector"]["final_loss_per_seq"]
    attnLoss = report["arms"]["attention"]["final_loss_per_seq"]
    attnArm = report["arms"]["attention"]
    markdown = [
        "# The bottleneck in miniature: a copy task", "",
        "Target == source over uniformly random tokens. Nothing is memorisable and",
        "there is no language model to lean on, so emitting target position *t*",
        "requires reading source position *t*. The only difference between the two",
        "arms is attention.", "",
        f"| arm | final loss / sequence | vs chance ({chanceLoss:.1f}) |", "|---|---|---|",
        f"| fixed vector (the paper's model) | **{fixedLoss:.3f}** | {fixedLoss / chanceLoss:.0%} |",
        f"| + Bahdanau attention | **{attnLoss:.3f}** | {attnLoss / chanceLoss:.0%} |", "",
        f"Alignment spread reached **{attnArm['alignment_spread']}** "
        f"(1.0 = fully peaked); the learned alignment is "
        f"{'the exact diagonal' if attnArm['is_diagonal'] else 'not diagonal'}.",
        "", "![Learned alignment](copy_task_alignment.png)", "",
        f"Optimiser: Adam, lr {args.lr}, a controlled demonstration rather than a "
        "reported reproduction. Plain SGD at the paper's lr 0.7 needs far more than "
        f"{args.steps} steps to leave chance on a {args.width}-unit model.",
    ]
    (outDir / "copy_task.md").write_text("\n".join(markdown) + "\n")
    print(f"[done] -> {outDir}")


if __name__ == "__main__":
    main()
