#!/usr/bin/env python3
"""Measure training throughput so the WMT subset size is chosen, not guessed.

Owner: M2. Reports target words/sec per device for a given model config, and
converts that into "hours per epoch" for a candidate corpus size.

`--arms base attn` measures both end-term arms with the same instrument and
prints the slowdown, which is how the attention arm's overnight slot is sized
instead of guessed. `--max-len 100` matches the end-term corpus, whose length
cap is 100 rather than 50.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seq2seq.data import ParallelDataset, collate, make_batches  # noqa: E402
from seq2seq.losses import chunked_ce_loss  # noqa: E402
from seq2seq.model import Seq2Seq  # noqa: E402
from seq2seq.vocab import SPECIALS, Vocab  # noqa: E402


def synthetic(n, vocab, mean_len=27, max_len=50, seed=0):
    """Length-distributed random pairs. `max_len` should match the corpus the
    estimate is for: the end-term data caps at 100, not 50, which raises the
    mean target length and therefore the cost of an epoch."""
    g = torch.Generator().manual_seed(seed)
    lens = torch.clamp(torch.normal(float(mean_len), 10.0, (n,), generator=g).long(), 4, max_len)
    src = [torch.randint(4, len(vocab), (int(L),), generator=g).tolist() for L in lens]
    tgt = [torch.randint(4, len(vocab), (int(L) + 2,), generator=g).tolist() for L in lens]
    return ParallelDataset(src, tgt)


def bench(device, vocab_size, emb, hid, layers, batch_size, steps, warmup=5,
          attention=None, mean_len=27, max_len=50, loss_chunk=0):
    vocab = Vocab(SPECIALS + [f"w{i}" for i in range(vocab_size - len(SPECIALS))])
    ds = synthetic(batch_size * (steps + warmup + 2), vocab,
                   mean_len=mean_len, max_len=max_len)
    dev = torch.device(device)
    model = Seq2Seq(len(vocab), len(vocab), emb_dim=emb, hidden_dim=hid,
                    num_layers=layers, src_pad_id=vocab.pad_id, tgt_pad_id=vocab.pad_id,
                    attention=attention, attn_dim=hid).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.7)
    crit = nn.CrossEntropyLoss(ignore_index=vocab.pad_id, reduction="sum")
    batches = make_batches(ds, batch_size, shuffle=True, seed=0)

    words = 0
    t0 = None
    for i, b in enumerate(batches[: steps + warmup]):
        src, src_len, tgt_in, tgt_out = collate(ds, b, vocab, vocab, True, dev)
        if loss_chunk:
            loss = chunked_ce_loss(model.forward_hidden(src, src_len, tgt_in),
                                   model.decoder.out, tgt_out, vocab.pad_id,
                                   loss_chunk) / len(b)
        else:
            loss = crit(model(src, src_len, tgt_in).reshape(-1, len(vocab)), tgt_out.reshape(-1)) / len(b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        if i == warmup - 1:
            t0, words = time.time(), 0
        if i >= warmup:
            words += int((tgt_out != vocab.pad_id).sum())
    return words / (time.time() - t0), model.count_parameters()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="+", default=None)
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--emb", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--corpus-pairs", type=int, default=500000)
    ap.add_argument("--mean-tgt-len", type=int, default=29)
    ap.add_argument("--arms", nargs="+", default=["base"], choices=["base", "attn"])
    ap.add_argument("--max-len", dest="max_len", type=int, default=50,
                    help="corpus length cap; the end-term data uses 100")
    ap.add_argument("--mean-len", dest="mean_len", type=int, default=27)
    ap.add_argument("--loss-chunk", dest="loss_chunk", type=int, default=0,
                    help="match the config's loss_chunk_size (WMT uses 8)")
    args = ap.parse_args()

    devices = args.devices
    if devices is None:
        devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
        if torch.cuda.is_available():
            devices = ["cuda"]

    print(f"model: {args.layers}x{args.hidden}, emb {args.emb}, vocab {args.vocab_size}, "
          f"batch {args.batch_size}, max_len {args.max_len}, loss_chunk {args.loss_chunk or 'off'}")
    epoch_words = args.corpus_pairs * args.mean_tgt_len
    rates = {}
    for arm in args.arms:
        attention = "bahdanau" if arm == "attn" else None
        for dev in devices:
            try:
                wps, params = bench(dev, args.vocab_size, args.emb, args.hidden,
                                    args.layers, args.batch_size, args.steps,
                                    attention=attention, mean_len=args.mean_len,
                                    max_len=args.max_len, loss_chunk=args.loss_chunk)
            except Exception as ex:  # a device that cannot run this config is not fatal
                print(f"  {arm:5s} {dev:5s}  FAILED: {type(ex).__name__}: {ex}")
                continue
            hours = epoch_words / wps / 3600
            print(f"  {arm:5s} {dev:5s}  {wps:8.0f} tgt-words/s  | {params/1e6:5.1f}M params "
                  f"| {hours:5.2f} h/epoch at {args.corpus_pairs:,} pairs "
                  f"| {8 * hours:5.1f} h for 8 epochs")
            rates[(arm, dev)] = wps

    # The number that sizes the overnight slot: how much the per-step attention
    # loop actually costs, measured rather than assumed.
    for dev in devices:
        if ("base", dev) in rates and ("attn", dev) in rates:
            ratio = rates[("base", dev)] / rates[("attn", dev)]
            print(f"\nattention slowdown on {dev}: {ratio:.2f}x "
                  f"({rates[('base', dev)]:.0f} -> {rates[('attn', dev)]:.0f} tgt-words/s)")
    if rates:
        arm, dev = max(rates, key=rates.get)
        print(f"fastest: {arm} on {dev} at {rates[(arm, dev)]:.0f} tgt-words/s")


if __name__ == "__main__":
    main()
