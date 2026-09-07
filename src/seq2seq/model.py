"""Encoder / decoder LSTMs -- the paper's architecture, scaled down.

Owner: M2 (model & training).

Faithful to Sutskever et al. (2014):
  * TWO SEPARATE LSTMs, no shared parameters (sec. 3.4: "32M for the encoder
    LSTM and 32M for the decoder LSTM").
  * The encoder's final (h, c) across all layers IS the sentence vector; the
    decoder is a language model initialised from it and conditioned on nothing
    else. This single fixed-length vector is exactly the bottleneck the
    end-term half of the project studies.
  * Uniform init U(-0.08, 0.08) on every parameter (sec. 3.4).
  * Naive full softmax over the target vocabulary (sec. 3.4).
  * No dropout in the paper; `dropout` defaults to 0.0 and any non-zero value
    is a declared deviation.

Scaled down: 2 layers x 512 cells / 512-d embeddings instead of 4 x 1000.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, pad_id, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, src, src_len):
        emb = self.embedding(src)
        packed = pack_padded_sequence(
            emb, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, c) = self.rnn(packed)
        return h, c  # the sentence vector: (num_layers, batch, hidden) x 2


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, pad_id, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_in, state):
        emb = self.embedding(tgt_in)
        output, state = self.rnn(emb, state)
        return self.out(output), state


class Seq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        emb_dim=512,
        hidden_dim=512,
        num_layers=2,
        src_pad_id=0,
        tgt_pad_id=0,
        dropout=0.0,
        init_range=0.08,
    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size, emb_dim, hidden_dim, num_layers, src_pad_id, dropout
        )
        self.decoder = Decoder(
            tgt_vocab_size, emb_dim, hidden_dim, num_layers, tgt_pad_id, dropout
        )
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.init_range = init_range
        self.reset_parameters()

    def reset_parameters(self):
        """Paper sec. 3.4: uniform between -0.08 and 0.08."""
        r = self.init_range
        for p in self.parameters():
            nn.init.uniform_(p, -r, r)

    def forward(self, src, src_len, tgt_in):
        state = self.encoder(src, src_len)
        logits, _ = self.decoder(tgt_in, state)
        return logits

    def encode(self, src, src_len):
        return self.encoder(src, src_len)

    @property
    def sentence_vector_size(self) -> int:
        """The paper's '8000 real numbers' -- num_layers x hidden x (h, c)."""
        return self.num_layers * self.hidden_dim * 2

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg, src_vocab, tgt_vocab) -> Seq2Seq:
    return Seq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        emb_dim=cfg["emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        dropout=cfg.get("dropout", 0.0),
        init_range=cfg.get("init_range", 0.08),
    )
