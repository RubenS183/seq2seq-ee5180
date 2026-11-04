"""Word-level vocabulary with the paper's UNK handling.

The paper keeps the 160k most frequent source and 80k target words and maps
every OOV to "UNK" (sec. 3.1). We keep the mechanism and shrink the size.
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
        self.stoi = {token: index for index, token in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.unk_id = self.stoi[UNK]
        self.sos_id = self.stoi[SOS]
        self.eos_id = self.stoi[EOS]

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, tokenStreams, maxSize: int, minFreq: int = 1) -> "Vocab":
        counts: Counter = Counter()
        for tokens in tokenStreams:
            counts.update(tokens)
        kept = [
            token
            for token, freq in counts.most_common()
            if freq >= minFreq and token not in SPECIALS
        ][: maxSize - len(SPECIALS)]
        return cls(SPECIALS + kept)

    def encode(self, tokens) -> list[int]:
        unkId, stoi = self.unk_id, self.stoi
        return [stoi.get(token, unkId) for token in tokens]

    def decode(self, ids, stripSpecials: bool = True) -> list[str]:
        decoded = []
        for tokenId in ids:
            token = self.itos[tokenId]
            if stripSpecials and token in (PAD, SOS, EOS):
                continue
            decoded.append(token)
        return decoded

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.itos, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text()))

    def unk_rate(self, tokenStreams) -> float:
        totalTokens = oovTokens = 0
        for tokens in tokenStreams:
            for token in tokens:
                totalTokens += 1
                oovTokens += token not in self.stoi
        return oovTokens / max(totalTokens, 1)
