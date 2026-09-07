"""Word-level vocabulary with the paper's UNK handling.

Paper uses 160k source / 80k target most-frequent words, every OOV -> "UNK"
(sec. 3.1). We keep the mechanism and shrink the size (declared deviation).

Owner: M1 (data).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"
SPECIALS = [PAD, UNK, SOS, EOS]


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = list(itos)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.unk_id = self.stoi[UNK]
        self.sos_id = self.stoi[SOS]
        self.eos_id = self.stoi[EOS]

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, token_streams, max_size: int, min_freq: int = 1) -> "Vocab":
        counter: Counter = Counter()
        for tokens in token_streams:
            counter.update(tokens)
        keep = [
            tok
            for tok, freq in counter.most_common()
            if freq >= min_freq and tok not in SPECIALS
        ][: max_size - len(SPECIALS)]
        return cls(SPECIALS + keep)

    def encode(self, tokens) -> list[int]:
        unk = self.unk_id
        stoi = self.stoi
        return [stoi.get(tok, unk) for tok in tokens]

    def decode(self, ids, strip_specials: bool = True) -> list[str]:
        out = []
        for i in ids:
            tok = self.itos[i]
            if strip_specials and tok in (PAD, SOS, EOS):
                continue
            out.append(tok)
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.itos, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text()))

    def unk_rate(self, token_streams) -> float:
        total = oov = 0
        for tokens in token_streams:
            for tok in tokens:
                total += 1
                oov += tok not in self.stoi
        return oov / max(total, 1)
