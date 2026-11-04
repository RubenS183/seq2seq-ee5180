"""Corpus cleaning, Moses tokenisation, source reversal, length-bucketed batching.

Two things here are load-bearing for the reproduction.

Source reversal (paper sec. 3.3). The source token sequence is reversed and the
target never is. We reverse the words and then append <eos>, so <eos> stays the
final encoder input in both arms and the sentence vector is always read off the
same event. The flag is applied identically at train and test time and it is the
ONLY difference between our two reported rows.

Length bucketing (paper sec. 3.4). Batches are built from sentences of similar
length, which the paper reports as a 2x speedup.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .vocab import Vocab


def _tokenizer(lang: str):
    from sacremoses import MosesTokenizer

    return MosesTokenizer(lang=lang)


def _tok_chunk(args):
    lang, lines = args
    tokenizer = _tokenizer(lang)
    return [tokenizer.tokenize(line, escape=False) for line in lines]


def tokenize_lines(lines: list[str], lang: str, workers: int = 0) -> list[list[str]]:
    """Moses-tokenise, preserving case, because the paper reports cased BLEU."""
    if workers and workers > 1 and len(lines) > 5000:
        import multiprocessing as mp

        chunkLen = (len(lines) + workers - 1) // workers
        pieces = [(lang, lines[i : i + chunkLen]) for i in range(0, len(lines), chunkLen)]
        with mp.Pool(workers) as pool:
            return [sentence for part in pool.map(_tok_chunk, pieces) for sentence in part]
    return _tok_chunk((lang, lines))


def detokenize(tokens: list[str], lang: str) -> str:
    from sacremoses import MosesDetokenizer

    return MosesDetokenizer(lang=lang).detokenize(tokens)


def clean_pairs(
    src: list[list[str]],
    tgt: list[list[str]],
    minLen: int = 1,
    maxLen: int = 50,
    maxRatio: float = 2.5,
    dedupe: bool = True,
) -> tuple[list[list[str]], list[list[str]], dict]:
    """Standard WMT-style filtering. Returns kept pairs plus a drop report."""
    keptSrc, keptTgt = [], []
    seen: set[tuple[str, str]] = set()
    stats = {"input": len(src), "empty": 0, "length": 0, "ratio": 0, "dupe": 0}
    for srcTokens, tgtTokens in zip(src, tgt):
        if not srcTokens or not tgtTokens:
            stats["empty"] += 1
            continue
        if not (minLen <= len(srcTokens) <= maxLen and minLen <= len(tgtTokens) <= maxLen):
            stats["length"] += 1
            continue
        if max(len(srcTokens), len(tgtTokens)) / min(len(srcTokens), len(tgtTokens)) > maxRatio:
            stats["ratio"] += 1
            continue
        if dedupe:
            key = (" ".join(srcTokens), " ".join(tgtTokens))
            if key in seen:
                stats["dupe"] += 1
                continue
            seen.add(key)
        keptSrc.append(srcTokens)
        keptTgt.append(tgtTokens)
    stats["kept"] = len(keptSrc)
    return keptSrc, keptTgt, stats


class Sequences:
    """Ragged integer sequences held as one flat array plus offsets.

    A list of 500k Python lists of small ints costs ~1 GB resident. The same data
    as int32 costs ~50 MB. That matters: at 1 GB the training process on a loaded
    16 GB machine gets its pages evicted and every minibatch pages back in, which
    measured 3x slower per step than the same code with the data resident.

    Indexing returns plain lists, so callers are unchanged.
    """

    __slots__ = ("flat", "offsets", "lengths")

    def __init__(self, flat, offsets):
        self.flat = flat
        self.offsets = offsets
        self.lengths = np.diff(offsets)

    @classmethod
    def from_lists(cls, sequences) -> "Sequences":
        lengths = np.fromiter(
            (len(s) for s in sequences), dtype=np.int64, count=len(sequences)
        )
        offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        flat = np.empty(int(offsets[-1]), dtype=np.int32)
        for index, sequence in enumerate(sequences):
            if sequence:
                flat[offsets[index] : offsets[index + 1]] = sequence
        return cls(flat, offsets)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        return self.flat[self.offsets[index] : self.offsets[index + 1]].tolist()

    def length(self, index) -> int:
        return int(self.lengths[index])

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


@dataclass
class ParallelDataset:
    src_ids: Sequences
    tgt_ids: Sequences

    def __post_init__(self):
        if not isinstance(self.src_ids, Sequences):
            self.src_ids = Sequences.from_lists(self.src_ids)
        if not isinstance(self.tgt_ids, Sequences):
            self.tgt_ids = Sequences.from_lists(self.tgt_ids)

    def __len__(self) -> int:
        return len(self.src_ids)

    @classmethod
    def encode(cls, srcToks, tgtToks, srcVocab: Vocab, tgtVocab: Vocab):
        return cls(
            [srcVocab.encode(s) for s in srcToks],
            [tgtVocab.encode(t) for t in tgtToks],
        )

    def save_npz(self, path) -> None:
        np.savez(
            path,
            src_flat=self.src_ids.flat, src_off=self.src_ids.offsets,
            tgt_flat=self.tgt_ids.flat, tgt_off=self.tgt_ids.offsets,
        )

    @classmethod
    def load_npz(cls, path) -> "ParallelDataset":
        stored = np.load(path)
        return cls(
            Sequences(stored["src_flat"], stored["src_off"]),
            Sequences(stored["tgt_flat"], stored["tgt_off"]),
        )


def make_batches(
    dataset: ParallelDataset,
    batchSize: int,
    shuffle: bool = True,
    seed: int = 0,
    bucketNoise: int = 1,
) -> list[list[int]]:
    """Length-bucketed minibatches (paper sec. 3.4).

    Sort by (source length, target length) with a little noise so epochs differ,
    cut into contiguous batches, then shuffle the batch ORDER. Sentences within a
    batch are near-equal length, so padding waste is small.
    """
    rng = random.Random(seed)
    order = list(range(len(dataset)))
    if shuffle:
        rng.shuffle(order)
    noise = (lambda: rng.randint(0, bucketNoise)) if (shuffle and bucketNoise) else (lambda: 0)
    srcLengths, tgtLengths = dataset.src_ids.lengths, dataset.tgt_ids.lengths
    order.sort(key=lambda i: (srcLengths[i] + noise(), tgtLengths[i]))
    batches = [order[i : i + batchSize] for i in range(0, len(order), batchSize)]
    if shuffle:
        rng.shuffle(batches)
    return batches


def collate(
    dataset: ParallelDataset,
    indices: list[int],
    srcVocab: Vocab,
    tgtVocab: Vocab,
    reverseSource: bool,
    device: torch.device | None = None,
    padMultiple: int = 8,
):
    """Build padded tensors for one minibatch.

    Encoder input : (reversed?) source words + <eos>
    Decoder input : <sos> + target words
    Decoder target: target words + <eos>
    """
    sources, targets = [], []
    for index in indices:
        sourceTokens = list(dataset.src_ids[index])
        if reverseSource:
            sourceTokens.reverse()
        sources.append(sourceTokens + [srcVocab.eos_id])
        targets.append(dataset.tgt_ids[index])

    batchSize = len(indices)
    srcWidth = max(len(s) for s in sources)
    tgtWidth = max(len(t) for t in targets) + 1  # +1 for <sos> / <eos>

    # Round the padded widths up to a multiple. Length bucketing otherwise yields
    # dozens of distinct tensor shapes per epoch, and Metal's caching allocator
    # keeps a block per shape, so memory creeps up until the machine swaps.
    # Quantising the shapes bounds that. It is purely padding: the encoder is
    # packed by true length and the extra target positions are padId, which the
    # loss ignores.
    if padMultiple > 1:
        srcWidth = -(-srcWidth // padMultiple) * padMultiple
        tgtWidth = -(-tgtWidth // padMultiple) * padMultiple

    src = torch.full((batchSize, srcWidth), srcVocab.pad_id, dtype=torch.long)
    srcLen = torch.empty(batchSize, dtype=torch.long)
    tgtIn = torch.full((batchSize, tgtWidth), tgtVocab.pad_id, dtype=torch.long)
    tgtOut = torch.full((batchSize, tgtWidth), tgtVocab.pad_id, dtype=torch.long)

    for row, (sourceTokens, targetTokens) in enumerate(zip(sources, targets)):
        src[row, : len(sourceTokens)] = torch.tensor(sourceTokens)
        srcLen[row] = len(sourceTokens)
        tgtIn[row, 0] = tgtVocab.sos_id
        tgtIn[row, 1 : len(targetTokens) + 1] = torch.tensor(targetTokens)
        tgtOut[row, : len(targetTokens)] = torch.tensor(targetTokens)
        tgtOut[row, len(targetTokens)] = tgtVocab.eos_id

    if device is not None:
        src, srcLen = src.to(device), srcLen.to(device)
        tgtIn, tgtOut = tgtIn.to(device), tgtOut.to(device)
    return src, srcLen, tgtIn, tgtOut


# Characters Python's universal-newline mode treats as line breaks but `wc -l`
# and the corpus authors do not. News-Commentary v9 contains ~2900 bare \r inside
# lines, and a DIFFERENT number of them in the English and French files.
# Splitting on those silently misaligns the parallel corpus, which is far worse
# than a crash because training would just quietly learn from mismatched pairs.
# So we split on "\n" only and neutralise the rest to spaces.
_STRAY_BREAKS = "\r\x0b\x0c\x1c\x1d\x1e\x85  "
_BREAK_TABLE = str.maketrans({c: " " for c in _STRAY_BREAKS})


def read_lines(path) -> list[str]:
    """Read a corpus file, splitting on newline ONLY.

    Returns exactly `wc -l` lines for a newline-terminated file.
    """
    with open(path, encoding="utf-8", newline="\n") as handle:
        text = handle.read()
    if text.endswith("\n"):
        text = text[:-1]
    if not text:
        return []
    return [line.translate(_BREAK_TABLE) for line in text.split("\n")]
