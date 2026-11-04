"""Bahdanau (additive) attention, the end-term's answer to the bottleneck.

Sutskever et al. push the whole source sentence through one fixed vector, so the
encoder must decide what to keep before it knows what the decoder will ask for.
Bahdanau, Cho & Bengio (2015) keep one annotation per source position and let
the decoder take a fresh weighted average of them at every step:

    e_tj = v^T tanh(W_enc h_j + W_dec s_{t-1})
    a_t  = softmax(e_t)        over source positions j
    c_t  = sum_j a_tj h_j
"""
from __future__ import annotations

import torch
import torch.nn as nn


def source_mask(srcLen: torch.Tensor, width: int) -> torch.Tensor:
    """True at padding positions. `srcLen` counts real tokens including <eos>.

    `data.collate` rounds the padded source width up to a multiple of 8, so a
    batch essentially always carries padding. Attending to it would be a silent
    bug, because the softmax would hand probability mass to embedded <pad>, so
    the mask is not optional.
    """
    positions = torch.arange(width, device=srcLen.device)
    return positions.unsqueeze(0) >= srcLen.unsqueeze(1)


class BahdanauAttention(nn.Module):
    """Additive attention. No biases, matching the paper's formulation."""

    def __init__(self, encDim: int, decDim: int, attnDim: int):
        super().__init__()
        self.W_enc = nn.Linear(encDim, attnDim, bias=False)
        self.W_dec = nn.Linear(decDim, attnDim, bias=False)
        self.v = nn.Linear(attnDim, 1, bias=False)

    def precompute(self, encOut: torch.Tensor) -> torch.Tensor:
        """W_enc h_j for every source position, once per sentence rather than
        once per (sentence, step) pair. Over ~30 decoder steps that is the
        difference between attention costing ~40% of a training step and
        costing several times the rest of the model."""
        return self.W_enc(encOut)

    def forward(self, decState, encOut, encProj, mask):
        """One decoder step's context vector and alignment.

        decState : (B, decDim)   the decoder's top-layer h BEFORE this step
        encOut   : (B, S, encDim)
        encProj  : (B, S, attnDim) from `precompute`
        mask     : (B, S) True at padding
        returns  : context (B, encDim), weights (B, S) summing to 1
        """
        energy = self.v(torch.tanh(encProj + self.W_dec(decState).unsqueeze(1)))
        energy = energy.squeeze(-1)
        if mask is not None:
            # -inf rather than a large negative constant, so the softmax returns
            # exactly 0.0 at padding. tests/test_attention.py asserts that.
            energy = energy.masked_fill(mask, float("-inf"))
        weights = torch.softmax(energy, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encOut).squeeze(1)
        return context, weights
