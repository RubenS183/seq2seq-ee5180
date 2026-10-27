#!/usr/bin/env python3
"""The bottleneck in miniature: a copy task both arms are asked to solve.

Owner: M4 (analysis & writing).

The end-term's claim about real translation takes ~11 GPU-hours to support. This
script makes the same point in about a minute, on a task with no confounds.

Target == source, over uniformly random tokens. Nothing can be memorised and
there is no language model to lean on, so emitting position t REQUIRES reading
source position t. A decoder that can only see a fixed-length summary of the
whole sequence has to store every token in it; attention just looks.

Measured result (seed 0, 1 layer x 64, 8 tokens): the attention arm reaches
near-zero loss and a clean diagonal alignment, while the fixed-vector arm is
still far from solving it. That is the bottleneck, isolated.

Adam, not the paper's SGD: this is a controlled demonstration, not a reported
reproduction, and plain SGD at lr 0.7 needs far longer to leave chance on a
model this small. Stated here rather than buried.
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


def alignment_spread(model, src, src_len, tgt_in) -> float:
    """Max-minus-min weight over REAL source positions (padding sits at 0.0)."""
    w = model.alignments(src, src_len, tgt_in)
    if w is None:
        return float("nan")
    pad = source_mask(src_len, src.size(1)).unsqueeze(1).expand_as(w)
    hi = w.masked_fill(pad, float("-inf")).amax(-1)
    lo = w.masked_fill(pad, float("inf")).amin(-1)
    return (hi - lo).max().item()


def run_arm(attention, vocab, length, batch, width, steps, lr, seed, log_every):
    torch.manual_seed(seed)
    model = Seq2Seq(len(vocab), len(vocab), emb_dim=width, hidden_dim=width,
                    num_layers=1, src_pad_id=vocab.pad_id, tgt_pad_id=vocab.pad_id,
                    attention=attention)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(ignore_index=vocab.pad_id, reduction="sum")
    gen = torch.Generator().manual_seed(seed)

    def make_batch():
        src = torch.randint(4, len(vocab), (batch, length), generator=gen)
        return (torch.cat([src, torch.full((batch, 1), vocab.eos_id)], 1),
                torch.full((batch,), length + 1),
                torch.cat([torch.full((batch, 1), vocab.sos_id), src], 1),
                torch.cat([src, torch.full((batch, 1), vocab.eos_id)], 1))

    curve = []
    src = src_len = tgt_in = None
    for step in range(1, steps + 1):
        src, src_len, tgt_in, tgt_out = make_batch()
        loss = crit(model(src, src_len, tgt_in).reshape(-1, len(vocab)),
                    tgt_out.reshape(-1)) / batch
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % log_every == 0 or step == 1:
            curve.append({"step": step, "loss_per_seq": round(loss.item(), 4)})
            print(f"    step {step:5d}  loss/seq {loss.item():8.3f}", flush=True)
    return model, curve, (src, src_len, tgt_in)


def main():
    ap = argparse.ArgumentParser(description="Copy-task bottleneck demonstration.")
    ap.add_argument("--out", default="results/copy_task")
    ap.add_argument("--length", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", dest="log_every", type=int, default=300)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vocab = Vocab(["<pad>", "<unk>", "<sos>", "<eos>"] + [f"w{i}" for i in range(20)])
    chance = (args.length + 1) * torch.tensor(float(len(vocab))).log().item()

    report = {"config": vars(args), "chance_loss_per_seq": round(chance, 3), "arms": {}}
    for name, attention in (("fixed vector", None), ("attention", "bahdanau")):
        print(f"  [{name}]", flush=True)
        model, curve, batch = run_arm(attention, vocab, args.length, args.batch,
                                      args.width, args.steps, args.lr, args.seed,
                                      args.log_every)
        entry = {"final_loss_per_seq": curve[-1]["loss_per_seq"], "curve": curve}
        if attention:
            src, src_len, tgt_in = batch
            entry["alignment_spread"] = round(alignment_spread(model, src, src_len, tgt_in), 4)
            weights = model.alignments(src, src_len, tgt_in)[0]
            peak = weights[: args.length].argmax(-1).tolist()
            entry["argmax_source_position"] = peak
            entry["is_diagonal"] = peak == list(range(args.length))
            toks = [f"s{i}" for i in range(args.length)] + ["<eos>"]
            plot_alignment_heatmap(
                weights.cpu().numpy(), toks, [f"t{i}" for i in range(args.length)] + ["<eos>"],
                out / "copy_task_alignment.png",
                "Copy task: learned alignment (attention arm)",
            )
        report["arms"][name] = entry
        print(f"    -> final loss/seq {entry['final_loss_per_seq']}  (chance {chance:.1f})\n", flush=True)

    (out / "copy_task.json").write_text(json.dumps(report, indent=2))

    fixed = report["arms"]["fixed vector"]["final_loss_per_seq"]
    attn = report["arms"]["attention"]["final_loss_per_seq"]
    md = [
        "# The bottleneck in miniature: a copy task", "",
        "Target == source over uniformly random tokens. Nothing is memorisable and",
        "there is no language model to lean on, so emitting target position *t*",
        "requires reading source position *t*. The only difference between the two",
        "arms is attention.", "",
        f"| arm | final loss / sequence | vs chance ({chance:.1f}) |", "|---|---|---|",
        f"| fixed vector (the paper's model) | **{fixed:.3f}** | {fixed / chance:.0%} |",
        f"| + Bahdanau attention | **{attn:.3f}** | {attn / chance:.0%} |", "",
        f"Alignment spread reached **{report['arms']['attention']['alignment_spread']}** "
        f"(1.0 = fully peaked); the learned alignment is "
        f"{'the exact diagonal' if report['arms']['attention']['is_diagonal'] else 'not diagonal'}.",
        "", "![Learned alignment](copy_task_alignment.png)", "",
        "Optimiser: Adam, lr "
        f"{args.lr} -- a controlled demonstration, not a reported reproduction. "
        "Plain SGD at the paper's lr 0.7 needs far more than "
        f"{args.steps} steps to leave chance on a {args.width}-unit model.",
    ]
    (out / "copy_task.md").write_text("\n".join(md) + "\n")
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
