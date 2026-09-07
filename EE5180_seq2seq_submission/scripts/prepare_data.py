#!/usr/bin/env python3
"""Tokenise, clean, build vocabularies and encode a parallel corpus.

Owner: M1 (data).

Only the TRAINING side is filtered (length / ratio / duplicates). Dev and test
are tokenised but never filtered -- dropping test sentences would inflate BLEU.
Raw (untokenised) references are kept alongside so we can report both a
detokenised sacreBLEU number and the paper-comparable tokenised one.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seq2seq.data import ParallelDataset, clean_pairs, read_lines, tokenize_lines  # noqa: E402
from seq2seq.vocab import Vocab  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory for prepared data")
    ap.add_argument("--src-lang", default="en")
    ap.add_argument("--tgt-lang", default="fr")
    ap.add_argument("--train-src", nargs="+", required=True)
    ap.add_argument("--train-tgt", nargs="+", required=True)
    ap.add_argument("--dev-src", required=True)
    ap.add_argument("--dev-tgt", required=True)
    ap.add_argument("--test-src", required=True)
    ap.add_argument("--test-tgt", required=True)
    ap.add_argument("--src-vocab-size", type=int, default=32000)
    ap.add_argument("--tgt-vocab-size", type=int, default=32000)
    ap.add_argument("--max-len", type=int, default=50)
    ap.add_argument("--max-ratio", type=float, default=2.5)
    ap.add_argument("--max-train-pairs", type=int, default=0, help="0 = keep all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"args": vars(args)}

    def tok(paths, lang):
        lines = []
        for p in paths if isinstance(paths, list) else [paths]:
            lines.extend(read_lines(p))
        print(f"  tokenising {len(lines):,} {lang} lines ...", flush=True)
        return tokenize_lines(lines, lang, workers=args.workers)

    print("[train]")
    tr_s = tok(args.train_src, args.src_lang)
    tr_t = tok(args.train_tgt, args.tgt_lang)
    assert len(tr_s) == len(tr_t), "train source/target line counts differ"

    tr_s, tr_t, stats = clean_pairs(
        tr_s, tr_t, max_len=args.max_len, max_ratio=args.max_ratio, dedupe=True
    )
    print(f"  clean: {stats}")
    report["clean"] = stats

    if args.max_train_pairs and len(tr_s) > args.max_train_pairs:
        rng = random.Random(args.seed)
        idx = rng.sample(range(len(tr_s)), args.max_train_pairs)
        idx.sort()
        tr_s = [tr_s[i] for i in idx]
        tr_t = [tr_t[i] for i in idx]
        print(f"  subsampled to {len(tr_s):,} pairs (seed {args.seed})")
    report["train_pairs"] = len(tr_s)

    print("[dev/test] (tokenised, NOT filtered)")
    dv_s, dv_t = tok(args.dev_src, args.src_lang), tok(args.dev_tgt, args.tgt_lang)
    te_s, te_t = tok(args.test_src, args.src_lang), tok(args.test_tgt, args.tgt_lang)

    print("[vocab]")
    src_vocab = Vocab.build(tr_s, args.src_vocab_size)
    tgt_vocab = Vocab.build(tr_t, args.tgt_vocab_size)
    src_vocab.save(out / "vocab.src.json")
    tgt_vocab.save(out / "vocab.tgt.json")
    report["src_vocab"] = len(src_vocab)
    report["tgt_vocab"] = len(tgt_vocab)
    report["unk_rate"] = {
        "train_src": src_vocab.unk_rate(tr_s), "train_tgt": tgt_vocab.unk_rate(tr_t),
        "test_src": src_vocab.unk_rate(te_s), "test_tgt": tgt_vocab.unk_rate(te_t),
    }
    print(f"  |src|={len(src_vocab):,} |tgt|={len(tgt_vocab):,}  unk rates: {report['unk_rate']}")

    for name, (s, t) in {"train": (tr_s, tr_t), "dev": (dv_s, dv_t), "test": (te_s, te_t)}.items():
        # .npz is what training loads: int32, ~10x smaller resident than the
        # JSON's Python int objects, which matters on a 16 GB machine.
        ParallelDataset(
            [src_vocab.encode(x) for x in s], [tgt_vocab.encode(x) for x in t]
        ).save_npz(out / f"{name}.npz")
        (out / f"{name}.tok.{args.src_lang}").write_text(
            "\n".join(" ".join(x) for x in s) + "\n", encoding="utf-8"
        )
        (out / f"{name}.tok.{args.tgt_lang}").write_text(
            "\n".join(" ".join(x) for x in t) + "\n", encoding="utf-8"
        )
        report[f"{name}_lines"] = len(s)

    # Raw references, needed for the detokenised sacreBLEU number.
    for split, path in (("dev", args.dev_tgt), ("test", args.test_tgt)):
        (out / f"{split}.raw.{args.tgt_lang}").write_text(
            "\n".join(read_lines(path)) + "\n", encoding="utf-8"
        )
    (out / f"test.raw.{args.src_lang}").write_text(
        "\n".join(read_lines(args.test_src)) + "\n", encoding="utf-8"
    )

    (out / "prepare_report.json").write_text(json.dumps(report, indent=2))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
