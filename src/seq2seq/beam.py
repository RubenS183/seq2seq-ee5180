"""Left-to-right beam search, configurable B, with ensemble support.

Faithful to paper sec. 3.2: each partial hypothesis is extended with every word,
all but the B most likely by LOG PROBABILITY are discarded, and a hypothesis
leaves the beam as soon as it emits <EOS>. So the beam genuinely shrinks as
hypotheses complete, and ranking uses raw log-probability with no length
normalisation. `lengthNorm` exists as a flag but every number we report uses the
paper's setting of 0.0. At B=1 this reduces exactly to greedy decoding, which
tests/test_smoke.py asserts.

Ensembling averages per-step log-probabilities across models, a geometric mean
of the member distributions. Members differ only in random init and minibatch
order, as in the paper.

Both end-term arms share this one implementation: everything carried between
steps lives in `model.DecodeState`, so the bookkeeping below never has to know
which architecture it is driving. A second copy would be a second place for
beam(B=1) to stop being greedy decoding.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_list(models):
    return models if isinstance(models, (list, tuple)) else [models]


def _encoder_input(srcIds, srcVocab, reverseSource, device):
    tokens = list(srcIds)
    if reverseSource:
        tokens.reverse()
    tokens = tokens + [srcVocab.eos_id]
    src = torch.tensor([tokens], dtype=torch.long, device=device)
    srcLen = torch.tensor([len(tokens)], dtype=torch.long, device=device)
    return src, srcLen


def _mean_step_logprobs(models, states, tokens):
    """Average next-token log-probabilities over the ensemble members."""
    total, nextStates = None, []
    for model, state in zip(models, states):
        logits, advanced = model.decode_step(tokens, state)
        logprobs = F.log_softmax(logits[:, -1, :].float(), dim=-1)
        total = logprobs if total is None else total + logprobs
        nextStates.append(advanced)
    return total / len(models), nextStates


@torch.no_grad()
def beam_search(
    models,
    srcIds: list[int],
    srcVocab,
    tgtVocab,
    beamSize: int = 12,
    reverseSource: bool = False,
    maxDecodeLen: int = 100,
    lengthNorm: float = 0.0,
    device=None,
):
    """Translate one sentence. Returns (tokenIds, logprob)."""
    models = _as_list(models)
    device = device or next(models[0].parameters()).device
    for model in models:
        model.eval()

    src, srcLen = _encoder_input(srcIds, srcVocab, reverseSource, device)

    # One decode state per model. It starts one hypothesis wide, and `reorder`
    # below grows it to the beam width on the first step.
    states = [model.encode_for_decode(src, srcLen) for model in models]

    beamTokens = torch.full((1, 1), tgtVocab.sos_id, dtype=torch.long, device=device)
    beamScores = torch.zeros(1, device=device)
    completed: list[tuple[float, list[int]]] = []
    maxSteps = min(maxDecodeLen, 2 * len(srcIds) + 5)

    for _ in range(maxSteps):
        if beamTokens.size(0) == 0:
            break

        stepLogprobs, nextStates = _mean_step_logprobs(
            models, states, beamTokens[:, -1:]
        )

        candidates = beamScores.unsqueeze(1) + stepLogprobs  # (beam, vocab)
        flat = candidates.reshape(-1)
        topK = min(beamSize, flat.numel())
        topScores, topFlatIdx = flat.topk(topK)
        beamIdx = torch.div(topFlatIdx, candidates.size(1), rounding_mode="floor")
        tokenIdx = topFlatIdx % candidates.size(1)

        nextTokens = torch.cat([beamTokens[beamIdx], tokenIdx.unsqueeze(1)], dim=1)

        # Paper: a hypothesis that emits <eos> leaves the beam.
        isEos = tokenIdx == tgtVocab.eos_id
        for row in torch.nonzero(isEos, as_tuple=False).flatten().tolist():
            sequence = nextTokens[row, 1:-1].tolist()  # drop <sos> and <eos>
            score = topScores[row].item()
            if lengthNorm:
                score = score / (max(len(sequence), 1) ** lengthNorm)
            completed.append((score, sequence))

        survived = ~isEos
        if survived.sum() == 0:
            break
        beamTokens = nextTokens[survived]
        beamScores = topScores[survived]
        # Reindexes the recurrent state and, for the attention arm, the memory
        # bank and mask alongside it.
        states = [state.reorder(beamIdx[survived]) for state in nextStates]

        # Admissible early stop: log-probabilities are negative, so extending a
        # partial hypothesis can only lower its score. Once the best partial is
        # already below the best completed hypothesis, nothing can overtake it.
        # Invalid under length normalisation, where completed scores are divided
        # by length and partials are not, so the comparison is gated on it.
        if not lengthNorm and len(completed) >= beamSize:
            if beamScores.max().item() < max(score for score, _ in completed):
                break

    if not completed:  # ran out of steps, so fall back to the best partial
        return beamTokens[0, 1:].tolist(), beamScores[0].item()

    completed.sort(key=lambda scored: scored[0], reverse=True)
    bestScore, bestSequence = completed[0]
    return bestSequence, bestScore


@torch.no_grad()
def greedy_decode(models, srcIds, srcVocab, tgtVocab, reverseSource=False,
                  maxDecodeLen=100, device=None):
    """Reference implementation used to assert that beam_search(B=1) == greedy.

    Deliberately does its own step arithmetic rather than calling the helper
    `beam_search` uses. A shared helper would make the two agree even when both
    are wrong, which is exactly what this gate is supposed to catch.
    """
    models = _as_list(models)
    device = device or next(models[0].parameters()).device
    src, srcLen = _encoder_input(srcIds, srcVocab, reverseSource, device)
    states = [model.encode_for_decode(src, srcLen) for model in models]
    token = torch.tensor([[tgtVocab.sos_id]], dtype=torch.long, device=device)
    emitted, score = [], 0.0
    for _ in range(min(maxDecodeLen, 2 * len(srcIds) + 5)):
        stepLogprobs, nextStates = None, []
        for model, state in zip(models, states):
            logits, advanced = model.decode_step(token, state)
            logprobs = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
            stepLogprobs = logprobs if stepLogprobs is None else stepLogprobs + logprobs
            nextStates.append(advanced)
        stepLogprobs = stepLogprobs / len(models)
        states = nextStates
        best = int(stepLogprobs.argmax(-1))
        score += float(stepLogprobs[0, best])
        if best == tgtVocab.eos_id:
            break
        emitted.append(best)
        token = torch.tensor([[best]], dtype=torch.long, device=device)
    return emitted, score
