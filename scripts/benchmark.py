#!/usr/bin/env python3
"""Measure training throughput so the WMT subset size is chosen, not guessed.

Reports target words/sec per device for a given model config and converts that
into hours per epoch for a candidate corpus size.

`--arms base attn` measures both end-term arms with the same instrument and
prints the slowdown, which is how the attention arm's overnight slot is sized
instead of guessed. `--max-len 100` matches the end-term corpus, whose length cap
is 100 rather than 50.
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


def synthetic(pairs, vocab, meanLen=27, maxLen=50, seed=0):
    """Length-distributed random pairs. `maxLen` should match the corpus the
    estimate is for: the end-term data caps at 100, not 50, which raises the mean
    target length and therefore the cost of an epoch."""
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.clamp(
        torch.normal(float(meanLen), 10.0, (pairs,), generator=generator).long(),
        4, maxLen,
    )
    src = [torch.randint(4, len(vocab), (int(n),), generator=generator).tolist()
           for n in lengths]
    tgt = [torch.randint(4, len(vocab), (int(n) + 2,), generator=generator).tolist()
           for n in lengths]
    return ParallelDataset(src, tgt)


def bench(device, vocabSize, embDim, hiddenDim, numLayers, batchSize, steps,
          warmup=5, attention=None, meanLen=27, maxLen=50, lossChunk=0):
    vocab = Vocab(SPECIALS + [f"w{i}" for i in range(vocabSize - len(SPECIALS))])
    dataset = synthetic(batchSize * (steps + warmup + 2), vocab,
                        meanLen=meanLen, maxLen=maxLen)
    torchDevice = torch.device(device)
    model = Seq2Seq(len(vocab), len(vocab), embDim=embDim, hiddenDim=hiddenDim,
                    numLayers=numLayers, srcPadId=vocab.pad_id,
                    tgtPadId=vocab.pad_id, attention=attention,
                    attnDim=hiddenDim).to(torchDevice)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.7)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id, reduction="sum")
    batches = make_batches(dataset, batchSize, shuffle=True, seed=0)

    targetWords, timedFrom = 0, None
    for step, batch in enumerate(batches[: steps + warmup]):
        src, srcLen, tgtIn, tgtOut = collate(
            dataset, batch, vocab, vocab, True, torchDevice
        )
        if lossChunk:
            loss = chunked_ce_loss(model.forward_hidden(src, srcLen, tgtIn),
                                   model.decoder.out, tgtOut, vocab.pad_id,
                                   lossChunk) / len(batch)
        else:
            logits = model(src, srcLen, tgtIn).reshape(-1, len(vocab))
            loss = criterion(logits, tgtOut.reshape(-1)) / len(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        if step == warmup - 1:
            timedFrom, targetWords = time.time(), 0
        if step >= warmup:
            targetWords += int((tgtOut != vocab.pad_id).sum())
    return targetWords / (time.time() - timedFrom), model.count_parameters()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--emb", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--corpus-pairs", type=int, default=500000)
    parser.add_argument("--mean-tgt-len", type=int, default=29)
    parser.add_argument("--arms", nargs="+", default=["base"], choices=["base", "attn"])
    parser.add_argument("--max-len", dest="max_len", type=int, default=50,
                        help="corpus length cap; the end-term data uses 100")
    parser.add_argument("--mean-len", dest="mean_len", type=int, default=27)
    parser.add_argument("--loss-chunk", dest="loss_chunk", type=int, default=0,
                        help="match the config's loss_chunk_size (WMT uses 8)")
    args = parser.parse_args()

    devices = args.devices
    if devices is None:
        devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
        if torch.cuda.is_available():
            devices = ["cuda"]

    print(f"model: {args.layers}x{args.hidden}, emb {args.emb}, "
          f"vocab {args.vocab_size}, batch {args.batch_size}, "
          f"max_len {args.max_len}, loss_chunk {args.loss_chunk or 'off'}")
    epochWords = args.corpus_pairs * args.mean_tgt_len
    rates = {}
    for arm in args.arms:
        attention = "bahdanau" if arm == "attn" else None
        for device in devices:
            try:
                wordsPerSec, params = bench(
                    device, args.vocab_size, args.emb, args.hidden, args.layers,
                    args.batch_size, args.steps, attention=attention,
                    meanLen=args.mean_len, maxLen=args.max_len,
                    lossChunk=args.loss_chunk,
                )
            except Exception as ex:  # a device that cannot run this config is not fatal
                print(f"  {arm:5s} {device:5s}  FAILED: {type(ex).__name__}: {ex}")
                continue
            hours = epochWords / wordsPerSec / 3600
            print(f"  {arm:5s} {device:5s}  {wordsPerSec:8.0f} tgt-words/s  "
                  f"| {params/1e6:5.1f}M params "
                  f"| {hours:5.2f} h/epoch at {args.corpus_pairs:,} pairs "
                  f"| {8 * hours:5.1f} h for 8 epochs")
            rates[(arm, device)] = wordsPerSec

    # The number that sizes the overnight slot: how much the per-step attention
    # loop actually costs, measured rather than assumed.
    for device in devices:
        if ("base", device) in rates and ("attn", device) in rates:
            ratio = rates[("base", device)] / rates[("attn", device)]
            print(f"\nattention slowdown on {device}: {ratio:.2f}x "
                  f"({rates[('base', device)]:.0f} -> "
                  f"{rates[('attn', device)]:.0f} tgt-words/s)")
    if rates:
        arm, device = max(rates, key=rates.get)
        print(f"fastest: {arm} on {device} at {rates[(arm, device)]:.0f} tgt-words/s")


if __name__ == "__main__":
    main()
