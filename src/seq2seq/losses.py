"""Memory-frugal cross-entropy for a large output vocabulary.

Owner: M2 (model & training).

The paper uses "a naive softmax over 80,000 words at each output" (sec. 3.4)
and parallelised it across four GPUs. At our scale the same naive softmax is
still the memory bottleneck: with batch 128, a 50-token target and a 32k
vocabulary the logits tensor alone is

    128 x 51 x 32000 x 4 bytes = 836 MB

and autograd retains it for the backward pass, alongside its gradient and the
softmax intermediates. That is what exhausted a 16 GB shared machine.

`chunked_ce_loss` splits the TIME axis into chunks and wraps each chunk's
projection + cross-entropy in `torch.utils.checkpoint`, so only one chunk's
logits exist at a time; the rest are recomputed during backward. Peak memory
for this term drops by roughly the number of chunks, at the cost of one extra
forward over the projection.

The maths is unchanged: the returned value and the gradients match the
unchunked computation to floating-point tolerance, which tests/test_smoke.py
asserts.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_nll(hidden_chunk, target_chunk, weight, bias, pad_id):
    """Projection + summed NLL for one slice of timesteps."""
    logits = F.linear(hidden_chunk, weight, bias)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_chunk.reshape(-1),
        ignore_index=pad_id,
        reduction="sum",
    )


def chunked_ce_loss(hidden, out_proj, targets, pad_id: int, chunk_size: int = 0):
    """Summed token NLL, computed in checkpointed chunks over time.

    hidden   : (batch, time, hidden_dim) decoder outputs, before projection
    out_proj : the nn.Linear mapping hidden -> vocabulary
    targets  : (batch, time) gold ids, padded with `pad_id`
    chunk_size: timesteps per chunk; 0 or >= time computes it in one go.
    """
    time_steps = hidden.size(1)
    if chunk_size <= 0 or chunk_size >= time_steps:
        logits = out_proj(hidden)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=pad_id,
            reduction="sum",
        )

    total = hidden.new_zeros(())
    use_ckpt = torch.is_grad_enabled() and hidden.requires_grad
    for start in range(0, time_steps, chunk_size):
        stop = min(start + chunk_size, time_steps)
        h, t = hidden[:, start:stop], targets[:, start:stop]
        if use_ckpt:
            total = total + checkpoint(
                _chunk_nll, h, t, out_proj.weight, out_proj.bias, pad_id,
                use_reentrant=False,
            )
        else:
            total = total + _chunk_nll(h, t, out_proj.weight, out_proj.bias, pad_id)
    return total
