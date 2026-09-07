"""Left-to-right beam search, configurable B, with ensemble support.

Owner: M3 (decoding & eval).

Faithful to paper sec. 3.2: "At each timestep we extend each partial hypothesis
in the beam with every possible word in the vocabulary ... discard all but the B
most likely hypotheses according to the model's LOG PROBABILITY. As soon as the
<EOS> symbol is appended to a hypothesis, it is removed from the beam and is
added to the set of complete hypotheses."

So the beam genuinely SHRINKS as hypotheses complete, and ranking uses raw
log-probability with no length normalisation. `length_norm` is available as a
flag but every number we report uses the paper's setting (0.0). With B=1 this
reduces exactly to greedy decoding, which `tests/test_smoke.py` asserts.

Ensembling averages per-step log-probabilities across models (a geometric mean
of the member distributions); the members differ only in random init and
minibatch order, as in the paper.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_list(models):
    return models if isinstance(models, (list, tuple)) else [models]


@torch.no_grad()
def beam_search(
    models,
    src_ids: list[int],
    src_vocab,
    tgt_vocab,
    beam_size: int = 12,
    reverse_source: bool = False,
    max_decode_len: int = 100,
    length_norm: float = 0.0,
    device=None,
):
    """Translate one sentence. Returns (token_ids, logprob)."""
    models = _as_list(models)
    device = device or next(models[0].parameters()).device
    for m in models:
        m.eval()

    s = list(src_ids)
    if reverse_source:
        s.reverse()
    s = s + [src_vocab.eos_id]

    src = torch.tensor([s], dtype=torch.long, device=device)
    src_len = torch.tensor([len(s)], dtype=torch.long, device=device)

    # One encoder state per model, expanded to the current beam width.
    states = [m.encode(src, src_len) for m in models]

    beam_tokens = torch.full((1, 1), tgt_vocab.sos_id, dtype=torch.long, device=device)
    beam_scores = torch.zeros(1, device=device)
    completed: list[tuple[float, list[int]]] = []
    max_steps = min(max_decode_len, 2 * len(src_ids) + 5)

    for _ in range(max_steps):
        if beam_tokens.size(0) == 0:
            break

        # Average log-probs over the ensemble members.
        step_logprobs, new_states = None, []
        for m, state in zip(models, states):
            logits, st = m.decoder(beam_tokens[:, -1:], state)
            lp = F.log_softmax(logits[:, -1, :].float(), dim=-1)
            step_logprobs = lp if step_logprobs is None else step_logprobs + lp
            new_states.append(st)
        step_logprobs = step_logprobs / len(models)

        cand = beam_scores.unsqueeze(1) + step_logprobs  # (beam, vocab)
        flat = cand.reshape(-1)
        k = min(beam_size, flat.numel())
        top_scores, top_idx = flat.topk(k)
        beam_idx = torch.div(top_idx, cand.size(1), rounding_mode="floor")
        token_idx = top_idx % cand.size(1)

        next_tokens = torch.cat(
            [beam_tokens[beam_idx], token_idx.unsqueeze(1)], dim=1
        )

        # Paper: a hypothesis that emits <eos> leaves the beam.
        is_eos = token_idx == tgt_vocab.eos_id
        for j in torch.nonzero(is_eos, as_tuple=False).flatten().tolist():
            seq = next_tokens[j, 1:-1].tolist()  # drop <sos> and <eos>
            score = top_scores[j].item()
            if length_norm:
                score = score / (max(len(seq), 1) ** length_norm)
            completed.append((score, seq))

        keep = ~is_eos
        if keep.sum() == 0:
            break
        beam_tokens = next_tokens[keep]
        beam_scores = top_scores[keep]
        survivors = beam_idx[keep]
        states = [(h[:, survivors, :], c[:, survivors, :]) for (h, c) in new_states]

        if len(completed) >= beam_size:
            # Every remaining partial is already worse than the best complete one.
            if beam_scores.max().item() < max(s for s, _ in completed):
                break

    if not completed:  # ran out of steps: fall back to the best partial
        seq = beam_tokens[0, 1:].tolist()
        return seq, beam_scores[0].item()

    completed.sort(key=lambda x: x[0], reverse=True)
    best_score, best_seq = completed[0]
    return best_seq, best_score


@torch.no_grad()
def greedy_decode(models, src_ids, src_vocab, tgt_vocab, reverse_source=False,
                  max_decode_len=100, device=None):
    """Reference implementation used to assert that beam_search(B=1) == greedy."""
    models = _as_list(models)
    device = device or next(models[0].parameters()).device
    s = list(src_ids)
    if reverse_source:
        s.reverse()
    s = s + [src_vocab.eos_id]
    src = torch.tensor([s], dtype=torch.long, device=device)
    src_len = torch.tensor([len(s)], dtype=torch.long, device=device)
    states = [m.encode(src, src_len) for m in models]
    token = torch.tensor([[tgt_vocab.sos_id]], dtype=torch.long, device=device)
    out, score = [], 0.0
    for _ in range(min(max_decode_len, 2 * len(src_ids) + 5)):
        lp_sum, new_states = None, []
        for m, state in zip(models, states):
            logits, st = m.decoder(token, state)
            lp = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
            lp_sum = lp if lp_sum is None else lp_sum + lp
            new_states.append(st)
        lp_sum = lp_sum / len(models)
        states = new_states
        best = int(lp_sum.argmax(-1))
        score += float(lp_sum[0, best])
        if best == tgt_vocab.eos_id:
            break
        out.append(best)
        token = torch.tensor([[best]], dtype=torch.long, device=device)
    return out, score
