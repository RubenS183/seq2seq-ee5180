"""Corpus cleaning, Moses tokenisation, source reversal, length-bucketed batching.

Owner: M1 (data).

Two things here are load-bearing for the reproduction:

*Source reversal* (paper sec. 3.3). The source token sequence is reversed, the
target never is. We reverse the words and then append <eos>, so <eos> stays the
final encoder input in both arms -- i.e. the sentence vector is always read off
the same event. The flag is applied identically at train and test time; it is
the ONLY difference between our two reported rows.

*Length bucketing* (paper sec. 3.4): batches are built from sentences of
similar length, which the paper reports as a 2x speedup.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch

from .vocab import Vocab


# ---------------------------------------------------------------- tokenisation

def _tokenizer(lang: str):
    from sacremoses import MosesTokenizer

    return MosesTokenizer(lang=lang)


def _tok_chunk(args):
    lang, lines = args
    tk = _tokenizer(lang)
    return [tk.tokenize(line, escape=False) for line in lines]


def tokenize_lines(lines: list[str], lang: str, workers: int = 0) -> list[list[str]]:
    """Moses-tokenise, preserving case (the paper reports *cased* BLEU)."""
    if workers and workers > 1 and len(lines) > 5000:
        import multiprocessing as mp

        chunk = (len(lines) + workers - 1) // workers
        pieces = [(lang, lines[i : i + chunk]) for i in range(0, len(lines), chunk)]
        with mp.Pool(workers) as pool:
            return [s for part in pool.map(_tok_chunk, pieces) for s in part]
    return _tok_chunk((lang, lines))


def detokenize(tokens: list[str], lang: str) -> str:
    from sacremoses import MosesDetokenizer

    return MosesDetokenizer(lang=lang).detokenize(tokens)


# -------------------------------------------------------------------- cleaning

def clean_pairs(
    src: list[list[str]],
    tgt: list[list[str]],
    min_len: int = 1,
    max_len: int = 50,
    max_ratio: float = 2.5,
    dedupe: bool = True,
) -> tuple[list[list[str]], list[list[str]], dict]:
    """Standard WMT-style filtering. Returns kept pairs plus a drop report."""
    keep_s, keep_t = [], []
    seen: set[tuple[str, str]] = set()
    stats = {"input": len(src), "empty": 0, "length": 0, "ratio": 0, "dupe": 0}
    for s, t in zip(src, tgt):
        if not s or not t:
            stats["empty"] += 1
            continue
        if not (min_len <= len(s) <= max_len and min_len <= len(t) <= max_len):
            stats["length"] += 1
            continue
        if max(len(s), len(t)) / min(len(s), len(t)) > max_ratio:
            stats["ratio"] += 1
            continue
        if dedupe:
            key = (" ".join(s), " ".join(t))
            if key in seen:
                stats["dupe"] += 1
                continue
            seen.add(key)
        keep_s.append(s)
        keep_t.append(t)
    stats["kept"] = len(keep_s)
    return keep_s, keep_t, stats


# --------------------------------------------------------------------- dataset

@dataclass
class ParallelDataset:
    src_ids: list[list[int]]
    tgt_ids: list[list[int]]

    def __len__(self) -> int:
        return len(self.src_ids)

    @classmethod
    def encode(cls, src_toks, tgt_toks, src_vocab: Vocab, tgt_vocab: Vocab):
        return cls(
            [src_vocab.encode(s) for s in src_toks],
            [tgt_vocab.encode(t) for t in tgt_toks],
        )


def make_batches(
    dataset: ParallelDataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
    bucket_noise: int = 1,
) -> list[list[int]]:
    """Length-bucketed minibatches (paper sec. 3.4).

    Sort by (source length, target length) with a little noise so epochs differ,
    cut into contiguous batches, then shuffle the batch *order*. Sentences within
    a batch are near-equal length, so padding waste is small.
    """
    rng = random.Random(seed)
    idx = list(range(len(dataset)))
    if shuffle:
        rng.shuffle(idx)
    noise = (lambda: rng.randint(0, bucket_noise)) if (shuffle and bucket_noise) else (lambda: 0)
    idx.sort(key=lambda i: (len(dataset.src_ids[i]) + noise(), len(dataset.tgt_ids[i])))
    batches = [idx[i : i + batch_size] for i in range(0, len(idx), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    return batches


def collate(
    dataset: ParallelDataset,
    indices: list[int],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    reverse_source: bool,
    device: torch.device | None = None,
):
    """Build padded tensors for one minibatch.

    Encoder input : (reversed?) source words + <eos>
    Decoder input : <sos> + target words
    Decoder target: target words + <eos>
    """
    srcs, tgts = [], []
    for i in indices:
        s = list(dataset.src_ids[i])
        if reverse_source:
            s.reverse()
        srcs.append(s + [src_vocab.eos_id])
        tgts.append(dataset.tgt_ids[i])

    bsz = len(indices)
    smax = max(len(s) for s in srcs)
    tmax = max(len(t) for t in tgts) + 1  # +1 for <sos> / <eos>

    src = torch.full((bsz, smax), src_vocab.pad_id, dtype=torch.long)
    src_len = torch.empty(bsz, dtype=torch.long)
    tgt_in = torch.full((bsz, tmax), tgt_vocab.pad_id, dtype=torch.long)
    tgt_out = torch.full((bsz, tmax), tgt_vocab.pad_id, dtype=torch.long)

    for b, (s, t) in enumerate(zip(srcs, tgts)):
        src[b, : len(s)] = torch.tensor(s)
        src_len[b] = len(s)
        tgt_in[b, 0] = tgt_vocab.sos_id
        tgt_in[b, 1 : len(t) + 1] = torch.tensor(t)
        tgt_out[b, : len(t)] = torch.tensor(t)
        tgt_out[b, len(t)] = tgt_vocab.eos_id

    if device is not None:
        src, src_len = src.to(device), src_len.to(device)
        tgt_in, tgt_out = tgt_in.to(device), tgt_out.to(device)
    return src, src_len, tgt_in, tgt_out


# Characters Python's universal-newline mode treats as line breaks but `wc -l`
# and the corpus authors do not. News-Commentary v9 contains ~2900 bare \r
# inside lines -- and a DIFFERENT number of them in the English and French
# files. Splitting on those silently misaligns the parallel corpus, which is
# far worse than a crash: training would just quietly learn from mismatched
# pairs. So we split on "\n" only and neutralise the rest to spaces.
_STRAY_BREAKS = "\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"
_BREAK_TABLE = str.maketrans({c: " " for c in _STRAY_BREAKS})


def read_lines(path) -> list[str]:
    """Read a corpus file, splitting on newline ONLY.

    Returns exactly `wc -l` lines for a newline-terminated file.
    """
    with open(path, encoding="utf-8", newline="\n") as fh:
        text = fh.read()
    if text.endswith("\n"):
        text = text[:-1]
    if not text:
        return []
    return [line.translate(_BREAK_TABLE) for line in text.split("\n")]
