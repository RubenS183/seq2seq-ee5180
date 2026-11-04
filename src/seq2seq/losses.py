"""Memory-frugal cross-entropy for a large output vocabulary.

The paper uses "a naive softmax over 80,000 words at each output" (sec. 3.4) and
parallelises it across four GPUs. At our scale that same softmax is still the
memory bottleneck: at batch 128, a 50-token target and a 32k vocabulary the
logits tensor alone is 128 x 51 x 32000 x 4 bytes = 836 MB, and autograd keeps
it alive for the backward pass.

`chunked_ce_loss` splits the time axis and wraps each chunk's projection and
cross-entropy in `torch.utils.checkpoint`, so only one chunk's logits exist at
once and the rest are recomputed during backward. The maths is unchanged and
tests/test_smoke.py asserts the value and every gradient match the unchunked
computation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_nll(hiddenChunk, targetChunk, weight, bias, padId):
    logits = F.linear(hiddenChunk, weight, bias)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targetChunk.reshape(-1),
        ignore_index=padId,
        reduction="sum",
    )


def chunked_ce_loss(hidden, outProj, targets, padId: int, chunkSize: int = 0):
    """Summed token NLL, computed in checkpointed chunks over time.

    hidden    : (batch, time, hiddenDim) decoder outputs, before projection
    outProj   : the nn.Linear mapping hidden -> vocabulary
    targets   : (batch, time) gold ids, padded with `padId`
    chunkSize : timesteps per chunk. 0 or >= time computes it in one go.
    """
    timeSteps = hidden.size(1)
    if chunkSize <= 0 or chunkSize >= timeSteps:
        logits = outProj(hidden)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=padId,
            reduction="sum",
        )

    total = hidden.new_zeros(())
    useCheckpoint = torch.is_grad_enabled() and hidden.requires_grad
    for start in range(0, timeSteps, chunkSize):
        stop = min(start + chunkSize, timeSteps)
        hiddenChunk, targetChunk = hidden[:, start:stop], targets[:, start:stop]
        if useCheckpoint:
            total = total + checkpoint(
                _chunk_nll, hiddenChunk, targetChunk,
                outProj.weight, outProj.bias, padId,
                use_reentrant=False,
            )
        else:
            total = total + _chunk_nll(
                hiddenChunk, targetChunk, outProj.weight, outProj.bias, padId
            )
    return total
