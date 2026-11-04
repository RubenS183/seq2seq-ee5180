#!/usr/bin/env python3
"""Deterministically subsample an aligned parallel corpus.

Used to cut Europarl down before tokenisation, because tokenising all 2.0M pairs
would hold several GB of Python strings for pairs we then throw away. Seeded, so
the subset is reproducible.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-in", required=True)
    parser.add_argument("--tgt-in", required=True)
    parser.add_argument("--src-out", required=True)
    parser.add_argument("--tgt-out", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    with open(args.src_in, encoding="utf-8") as handle:
        totalLines = sum(1 for _ in handle)
    if args.n >= totalLines:
        keptRows = None
        print(f"[subsample] {args.src_in}: {totalLines:,} <= n, copying all")
    else:
        rng = random.Random(args.seed)
        keptRows = set(rng.sample(range(totalLines), args.n))
        print(f"[subsample] {totalLines:,} -> {args.n:,} pairs (seed {args.seed})")

    for inPath, outPath in ((args.src_in, args.src_out), (args.tgt_in, args.tgt_out)):
        Path(outPath).parent.mkdir(parents=True, exist_ok=True)
        with open(inPath, encoding="utf-8") as fin, \
             open(outPath, "w", encoding="utf-8") as fout:
            for row, line in enumerate(fin):
                if keptRows is None or row in keptRows:
                    fout.write(line)


if __name__ == "__main__":
    main()
