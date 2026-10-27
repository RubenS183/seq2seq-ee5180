"""Bahdanau (additive) attention -- the end-term's answer to the bottleneck.

Owner: M2 (model & training).

Sutskever et al. push the whole source sentence through one fixed vector `v`
(4 layers x 1000 cells = 8000 numbers in the paper, 2 x 512 x 2 = 2048 here).
Every decoder step reads the same `v`, so the encoder must decide what to keep
before it knows what the decoder will ask for. Bahdanau, Cho & Bengio (2015)
remove that constraint: the encoder keeps one annotation per source position and
the decoder computes a fresh weighted average of them at every step.

We implement the paper's *additive* ("concat") energy exactly:

    e_tj = v^T tanh(W_enc h_j + W_dec s_{t-1})
    a_t  = softmax(e_t)                           (over source positions j)
    c_t  = sum_j a_tj h_j

`W_enc h_j` does not depend on the decoder step, so `precompute` applies it once
per sentence instead of once per (sentence, step) pair. With ~30 decoder steps
that is the difference between attention costing ~40% of a training step and
costing several times the rest of the model.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def source_mask(src_len: torch.Tensor, width: int) -> torch.Tensor:
    """True at padding positions. `src_len` counts real tokens incl. <eos>.

    `data.collate` rounds the padded source width up to a multiple of 8, so a
    batch essentially always carries padding. Attending to it would be a silent
    bug -- the softmax would hand probability mass to embedded <pad> -- so the
    mask is not optional and is derived from the `src_len` the collate function
    already returns everywhere.
    """
    positions = torch.arange(width, device=src_len.device)
    return positions.unsqueeze(0) >= src_len.unsqueeze(1)


class BahdanauAttention(nn.Module):
    """Additive attention. No biases, matching the paper's formulation."""

    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def precompute(self, enc_out: torch.Tensor) -> torch.Tensor:
        """W_enc h_j for every source position, once per sentence. (B, S, A)"""
        return self.W_enc(enc_out)

    def forward(self, dec_state, enc_out, enc_proj, mask):
        """One decoder step's context vector and alignment.

        dec_state : (B, dec_dim)  the decoder's top-layer h BEFORE this step
        enc_out   : (B, S, enc_dim)
        enc_proj  : (B, S, attn_dim) from `precompute`
        mask      : (B, S) True at padding
        returns   : context (B, enc_dim), weights (B, S) summing to 1
        """
        energy = self.v(torch.tanh(enc_proj + self.W_dec(dec_state).unsqueeze(1)))
        energy = energy.squeeze(-1)  # (B, S)
        if mask is not None:
            # -inf, not a large negative constant: softmax then returns exactly
            # 0.0 at padding, which tests/test_attention.py asserts.
            energy = energy.masked_fill(mask, float("-inf"))
        weights = torch.softmax(energy, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)
        return context, weights
