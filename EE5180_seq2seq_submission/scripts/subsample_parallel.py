#!/usr/bin/env python3
"""Deterministically subsample an aligned parallel corpus.

Owner: M1. Used to cut Europarl down before tokenisation -- tokenising all
2.0M pairs would hold several GB of Python strings for pairs we then throw
away. Seeded, so the subset is reproducible.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-in", required=True)
    ap.add_argument("--tgt-in", required=True)
    ap.add_argument("--src-out", required=True)
    ap.add_argument("--tgt-out", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    with open(args.src_in, encoding="utf-8") as fh:
        total = sum(1 for _ in fh)
    if args.n >= total:
        keep = None
        print(f"[subsample] {args.src_in}: {total:,} <= n, copying all")
    else:
        rng = random.Random(args.seed)
        keep = set(rng.sample(range(total), args.n))
        print(f"[subsample] {total:,} -> {args.n:,} pairs (seed {args.seed})")

    for src_in, out in ((args.src_in, args.src_out), (args.tgt_in, args.tgt_out)):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(src_in, encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                if keep is None or i in keep:
                    fout.write(line)


if __name__ == "__main__":
    main()
