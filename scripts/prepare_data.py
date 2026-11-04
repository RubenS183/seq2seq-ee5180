#!/usr/bin/env python3
"""Tokenise, clean, build vocabularies and encode a parallel corpus.

Only the TRAINING side is filtered on length, ratio and duplicates. Dev and test
are tokenised but never filtered, because dropping test sentences would inflate
BLEU. Raw untokenised references are kept alongside so we can report both a
detokenised sacreBLEU number and the paper-comparable tokenised one.

`--max-ratio inf` disables length-mismatch filtering entirely, which is how the
end-term data is built. The drop report then records `"ratio": 0`, which is the
artefact evidencing that no pair was dropped for a source/target length mismatch.
The mid-term defaults (max_len 50, max_ratio 2.5) are left alone so `make
data-wmt` keeps reproducing the submitted corpus exactly.
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="output directory for prepared data")
    parser.add_argument("--src-lang", default="en")
    parser.add_argument("--tgt-lang", default="fr")
    parser.add_argument("--train-src", nargs="+", required=True)
    parser.add_argument("--train-tgt", nargs="+", required=True)
    parser.add_argument("--dev-src", required=True)
    parser.add_argument("--dev-tgt", required=True)
    parser.add_argument("--test-src", required=True)
    parser.add_argument("--test-tgt", required=True)
    parser.add_argument("--src-vocab-size", type=int, default=32000)
    parser.add_argument("--tgt-vocab-size", type=int, default=32000)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--max-ratio", type=float, default=2.5)
    parser.add_argument("--max-train-pairs", type=int, default=0, help="0 = keep all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    outDir = Path(args.out).expanduser()
    outDir.mkdir(parents=True, exist_ok=True)
    # json.dumps writes a bare `Infinity` for float("inf"), which is not valid
    # JSON and breaks any strict reader of prepare_report.json later, so record
    # it as the string "inf" instead.
    report: dict = {
        "args": {
            key: ("inf" if isinstance(value, float) and value == float("inf") else value)
            for key, value in vars(args).items()
        }
    }

    def tokenizeAll(paths, lang):
        lines = []
        for path in paths if isinstance(paths, list) else [paths]:
            lines.extend(read_lines(path))
        print(f"  tokenising {len(lines):,} {lang} lines ...", flush=True)
        return tokenize_lines(lines, lang, workers=args.workers)

    print("[train]")
    trainSrc = tokenizeAll(args.train_src, args.src_lang)
    trainTgt = tokenizeAll(args.train_tgt, args.tgt_lang)
    assert len(trainSrc) == len(trainTgt), "train source/target line counts differ"

    trainSrc, trainTgt, dropStats = clean_pairs(
        trainSrc, trainTgt, maxLen=args.max_len, maxRatio=args.max_ratio, dedupe=True
    )
    print(f"  clean: {dropStats}")
    report["clean"] = dropStats

    if args.max_train_pairs and len(trainSrc) > args.max_train_pairs:
        rng = random.Random(args.seed)
        kept = sorted(rng.sample(range(len(trainSrc)), args.max_train_pairs))
        trainSrc = [trainSrc[i] for i in kept]
        trainTgt = [trainTgt[i] for i in kept]
        print(f"  subsampled to {len(trainSrc):,} pairs (seed {args.seed})")
    report["train_pairs"] = len(trainSrc)

    print("[dev/test] (tokenised, NOT filtered)")
    devSrc = tokenizeAll(args.dev_src, args.src_lang)
    devTgt = tokenizeAll(args.dev_tgt, args.tgt_lang)
    testSrc = tokenizeAll(args.test_src, args.src_lang)
    testTgt = tokenizeAll(args.test_tgt, args.tgt_lang)

    print("[vocab]")
    srcVocab = Vocab.build(trainSrc, args.src_vocab_size)
    tgtVocab = Vocab.build(trainTgt, args.tgt_vocab_size)
    srcVocab.save(outDir / "vocab.src.json")
    tgtVocab.save(outDir / "vocab.tgt.json")
    report["src_vocab"] = len(srcVocab)
    report["tgt_vocab"] = len(tgtVocab)
    report["unk_rate"] = {
        "train_src": srcVocab.unk_rate(trainSrc),
        "train_tgt": tgtVocab.unk_rate(trainTgt),
        "test_src": srcVocab.unk_rate(testSrc),
        "test_tgt": tgtVocab.unk_rate(testTgt),
    }
    print(f"  |src|={len(srcVocab):,} |tgt|={len(tgtVocab):,}  "
          f"unk rates: {report['unk_rate']}")

    splits = {
        "train": (trainSrc, trainTgt),
        "dev": (devSrc, devTgt),
        "test": (testSrc, testTgt),
    }
    for name, (srcTokens, tgtTokens) in splits.items():
        # .npz is what training loads: int32, ~10x smaller resident than the
        # JSON's Python int objects, which matters on a 16 GB machine.
        ParallelDataset(
            [srcVocab.encode(s) for s in srcTokens],
            [tgtVocab.encode(t) for t in tgtTokens],
        ).save_npz(outDir / f"{name}.npz")
        (outDir / f"{name}.tok.{args.src_lang}").write_text(
            "\n".join(" ".join(s) for s in srcTokens) + "\n", encoding="utf-8"
        )
        (outDir / f"{name}.tok.{args.tgt_lang}").write_text(
            "\n".join(" ".join(t) for t in tgtTokens) + "\n", encoding="utf-8"
        )
        report[f"{name}_lines"] = len(srcTokens)

    # Raw references, needed for the detokenised sacreBLEU number.
    for split, path in (("dev", args.dev_tgt), ("test", args.test_tgt)):
        (outDir / f"{split}.raw.{args.tgt_lang}").write_text(
            "\n".join(read_lines(path)) + "\n", encoding="utf-8"
        )
    (outDir / f"test.raw.{args.src_lang}").write_text(
        "\n".join(read_lines(args.test_src)) + "\n", encoding="utf-8"
    )

    (outDir / "prepare_report.json").write_text(json.dumps(report, indent=2))
    print(f"[done] -> {outDir}")


if __name__ == "__main__":
    main()
