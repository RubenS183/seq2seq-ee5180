"""Correctness gates that must pass before any long training run.

The MPS check matters: PyTorch's Metal LSTM has historically produced silently
wrong numbers, and discovering that after an overnight run costs a day.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seq2seq.beam import beam_search, greedy_decode  # noqa: E402
from seq2seq.data import ParallelDataset, clean_pairs, collate, make_batches  # noqa: E402
from seq2seq.model import Seq2Seq  # noqa: E402
from seq2seq.vocab import Vocab  # noqa: E402


def tiny_vocabs():
    vocab = Vocab(["<pad>", "<unk>", "<sos>", "<eos>"] + [f"w{i}" for i in range(20)])
    return vocab, vocab


def tiny_model(seed=0):
    torch.manual_seed(seed)
    srcVocab, tgtVocab = tiny_vocabs()
    return Seq2Seq(len(srcVocab), len(tgtVocab), embDim=16, hiddenDim=16, numLayers=2,
                   srcPadId=srcVocab.pad_id, tgtPadId=tgtVocab.pad_id), srcVocab, tgtVocab


# data layer

def test_init_range_is_the_papers():
    model, _, _ = tiny_model()
    w = torch.cat([path.flatten() for path in model.parameters()]).detach()
    assert w.min() >= -0.08 and w.max() <= 0.08


def test_reversal_reverses_source_only_and_keeps_eos_last():
    srcVocab, tgtVocab = tiny_vocabs()
    dataset = ParallelDataset([[4, 5, 6]], [[7, 8]])
    fwd = collate(dataset, [0], srcVocab, tgtVocab, reverseSource=False, padMultiple=1)
    rev = collate(dataset, [0], srcVocab, tgtVocab, reverseSource=True, padMultiple=1)
    assert fwd[0][0].tolist() == [4, 5, 6, srcVocab.eos_id]
    assert rev[0][0].tolist() == [6, 5, 4, srcVocab.eos_id]      # words reversed, <eos> still last
    assert fwd[3][0].tolist() == rev[3][0].tolist()        # target untouched


def test_collate_targets_are_shifted_by_one():
    srcVocab, tgtVocab = tiny_vocabs()
    dataset = ParallelDataset([[4, 5]], [[7, 8, 9]])
    _, _, tgt_in, tgt_out = collate(dataset, [0], srcVocab, tgtVocab, reverseSource=False, padMultiple=1)
    assert tgt_in[0].tolist() == [tgtVocab.sos_id, 7, 8, 9]
    assert tgt_out[0].tolist() == [7, 8, 9, tgtVocab.eos_id]


def test_shape_quantisation_pads_only_and_changes_no_loss():
    """Rounding padded lengths up to a multiple must be inert.

    Shapes get bigger, the content up to the true length is identical, and the
    loss is unchanged because the extra target positions are pad_id (ignored)
    and the encoder is packed by true length.
    """
    from seq2seq.losses import chunked_ce_loss

    model, srcVocab, tgtVocab = tiny_model(seed=31)
    dataset = ParallelDataset([[4, 5, 6], [7, 8]], [[9, 10], [11, 12, 13]])

    exact = collate(dataset, [0, 1], srcVocab, tgtVocab, reverseSource=True, padMultiple=1)
    padded = collate(dataset, [0, 1], srcVocab, tgtVocab, reverseSource=True, padMultiple=8)

    assert padded[0].size(1) % 8 == 0 and padded[2].size(1) % 8 == 0
    assert padded[0].size(1) >= exact[0].size(1)
    assert torch.equal(exact[1], padded[1])                     # true lengths identical
    n = exact[0].size(1)
    assert torch.equal(exact[0], padded[0][:, :n])              # same content
    assert (padded[0][:, n:] == srcVocab.pad_id).all()                # remainder is padding

    with torch.no_grad():
        losses = []
        for src, src_len, tgt_in, tgt_out in (exact, padded):
            h = model.forward_hidden(src, src_len, tgt_in)
            losses.append(chunked_ce_loss(h, model.decoder.out, tgt_out, tgtVocab.pad_id, 0).item())
    assert abs(losses[0] - losses[1]) < 1e-4, f"padding changed the loss: {losses}"


def test_length_bucketing_groups_similar_lengths():
    dataset = ParallelDataset([[4] * (i % 40 + 1) for i in range(512)],
                         [[5] * (i % 40 + 1) for i in range(512)])
    batches = make_batches(dataset, 32, shuffle=True, seed=0)
    assert sum(len(b) for b in batches) == 512
    spread = max(max(len(dataset.src_ids[i]) for i in b) - min(len(dataset.src_ids[i]) for i in b)
                 for b in batches)
    assert spread <= 4, f"buckets too loose (spread {spread})"


def test_clean_pairs_filters_and_reports():
    s = [["a"], [], ["b"] * 60, ["c"], ["c"], ["d"]]
    t = [["x"], ["y"], ["z"], ["w"], ["w"], ["e"] * 5]
    keptSrc, keptTgt, state = clean_pairs(s, t, maxLen=50, maxRatio=2.5)
    assert state["empty"] == 1 and state["length"] == 1 and state["dupe"] == 1 and state["ratio"] == 1
    assert len(keptSrc) == len(keptTgt) == 2


# model

def test_forward_shapes():
    model, srcVocab, tgtVocab = tiny_model()
    src = torch.randint(4, 20, (3, 7)); src_len = torch.tensor([7, 5, 6])
    logits = model(src, src_len, torch.randint(4, 20, (3, 9)))
    assert logits.shape == (3, 9, len(tgtVocab))


def test_padding_does_not_change_the_sentence_vector():
    """Encoding with extra padding must equal encoding the bare sentence."""
    model, srcVocab, tgtVocab = tiny_model()
    model.eval()
    sequence = [4, 5, 6]
    with torch.no_grad():
        h1, c1 = model.encode(torch.tensor([sequence]), torch.tensor([3]))
        padded = torch.tensor([sequence + [srcVocab.pad_id] * 4])
        h2, c2 = model.encode(padded, torch.tensor([3]))
    assert torch.allclose(h1, h2, atol=1e-6) and torch.allclose(c1, c2, atol=1e-6)


def test_model_can_overfit_a_tiny_batch():
    """If the training signal is wired up correctly this drives loss to ~0."""
    torch.manual_seed(0)
    model, srcVocab, tgtVocab = tiny_model()
    dataset = ParallelDataset([[4, 5, 6], [7, 8]], [[9, 10], [11, 12, 13]])
    src, src_len, tgt_in, tgt_out = collate(dataset, [0, 1], srcVocab, tgtVocab, reverseSource=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    crit = torch.nn.CrossEntropyLoss(ignore_index=tgtVocab.pad_id, reduction="sum")
    first = last = None
    for i in range(300):
        loss = crit(model(src, src_len, tgt_in).reshape(-1, len(tgtVocab)), tgt_out.reshape(-1)) / 2
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        first = loss.item() if i == 0 else first
        last = loss.item()
    assert last < 0.05 * first, f"failed to overfit: {first:.3f} -> {last:.3f}"


# beam search

def test_beam_one_equals_greedy():
    model, srcVocab, tgtVocab = tiny_model(seed=3)
    for src in ([4, 5, 6], [7], [8, 9, 10, 11, 12]):
        b, _ = beam_search(model, src, srcVocab, tgtVocab, beamSize=1, reverseSource=True)
        g, _ = greedy_decode(model, src, srcVocab, tgtVocab, reverseSource=True)
        assert b == g, f"beam(B=1) != greedy for {src}: {b} vs {g}"


def test_wider_beam_never_finds_a_worse_hypothesis():
    model, srcVocab, tgtVocab = tiny_model(seed=5)
    for src in ([4, 5, 6], [8, 9, 10, 11]):
        _, score1 = beam_search(model, src, srcVocab, tgtVocab, beamSize=1, reverseSource=True)
        _, score12 = beam_search(model, src, srcVocab, tgtVocab, beamSize=12, reverseSource=True)
        assert score12 >= score1 - 1e-5, f"B=12 scored worse than B=1: {score12} < {score1}"


def test_beam_respects_reversal_flag():
    model, srcVocab, tgtVocab = tiny_model(seed=7)
    a, _ = beam_search(model, [4, 5, 6, 7], srcVocab, tgtVocab, beamSize=4, reverseSource=False)
    b, _ = beam_search(model, [7, 6, 5, 4], srcVocab, tgtVocab, beamSize=4, reverseSource=True)
    assert a == b, "reversing the input and setting the flag must cancel out"


# BLEU gate

def test_bleu_of_references_against_themselves_is_100():
    from seq2seq.evaluate import score_tokenized

    refs = ["le chat est sur le tapis", "un homme marche dans la rue"]
    score, _ = score_tokenized(refs, refs)
    assert score == pytest.approx(100.0, abs=1e-6)


# device agreement

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_mps_matches_cpu_forward_and_backward():
    """Guards against silently wrong Metal LSTM kernels."""
    torch.manual_seed(11)
    model, srcVocab, tgtVocab = tiny_model(seed=11)
    dataset = ParallelDataset([[4, 5, 6], [7, 8], [9, 10, 11, 12]], [[13, 14], [15], [16, 17, 18]])
    crit = torch.nn.CrossEntropyLoss(ignore_index=tgtVocab.pad_id, reduction="sum")

    def loss_and_grad(device):
        otherModel = Seq2Seq(len(srcVocab), len(tgtVocab), embDim=16, hiddenDim=16, numLayers=2,
                     srcPadId=srcVocab.pad_id, tgtPadId=tgtVocab.pad_id)
        otherModel.load_state_dict(model.state_dict())
        otherModel = otherModel.to(device)
        src, src_len, tgt_in, tgt_out = collate(dataset, [0, 1, 2], srcVocab, tgtVocab, True, torch.device(device))
        loss = crit(otherModel(src, src_len, tgt_in).reshape(-1, len(tgtVocab)), tgt_out.reshape(-1)) / 3
        loss.backward()
        grad = torch.cat([path.grad.flatten().cpu() for path in otherModel.parameters() if path.grad is not None])
        return loss.item(), grad

    lc, gc = loss_and_grad("cpu")
    lm, gm = loss_and_grad("mps")
    assert abs(lc - lm) < 1e-3, f"loss mismatch cpu={lc} mps={lm}"
    assert (gc - gm).abs().max() < 1e-3, f"grad mismatch {(gc - gm).abs().max()}"


# corpus reading (alignment)

def test_read_lines_splits_on_newline_only(tmp_path):
    """Regression: News-Commentary v9 carries ~2900 bare \\r *inside* lines, and a
    different count per language. Python's universal-newline mode split on them
    and silently misaligned the parallel corpus."""
    from seq2seq.data import read_lines

    path = tmp_path / "corpus.txt"
    path.write_bytes(
        "one\rwith carriage return\n"
        "two with unicode separator\n"
        "three\n".encode("utf-8")
    )
    lines = read_lines(path)
    assert len(lines) == 3, f"expected 3 lines, got {len(lines)}: {lines}"
    assert "\r" not in "".join(lines) and " " not in "".join(lines)
    assert lines[2] == "three"


def test_read_lines_keeps_parallel_files_aligned(tmp_path):
    from seq2seq.data import read_lines

    src = tmp_path / "a.en"
    tgt = tmp_path / "a.fr"
    src.write_bytes("x\ry\nz\n".encode())      # one stray \r
    tgt.write_bytes("p\nq\n".encode())          # none
    assert len(read_lines(src)) == len(read_lines(tgt)) == 2


def test_wider_beam_not_worse_under_length_normalisation():
    """Beam monotonicity must also hold on the --length-norm path.

    Note on strength: this is a property check, not a regression test for the
    early-stop guard in beam.py. On a tiny random model the guarded comparison
    happens not to change the outcome, so this test still passes if the guard is
    removed. The guard stays because the comparison is invalid on its face --
    completed hypotheses carry length-normalised scores while partials do not --
    not because this test catches it.
    """
    model, srcVocab, tgtVocab = tiny_model(seed=13)
    for src in ([4, 5, 6], [7, 8, 9, 10], [11, 12]):
        _, score2 = beam_search(model, src, srcVocab, tgtVocab, beamSize=2, reverseSource=True, lengthNorm=1.0)
        _, score12 = beam_search(model, src, srcVocab, tgtVocab, beamSize=12, reverseSource=True, lengthNorm=1.0)
        assert score12 >= score2 - 1e-5, f"length-normalised B=12 scored worse than B=2 on {src}: {score12} < {score2}"


# compact dataset storage

def test_sequences_round_trip_exactly():
    from seq2seq.data import Sequences

    raw = [[4, 5, 6], [7], [], [8, 9, 10, 11], [12, 13]]
    sequence = Sequences.from_lists(raw)
    assert len(sequence) == len(raw)
    assert [sequence[i] for i in range(len(raw))] == raw
    assert sequence[1:4] == raw[1:4]
    assert sequence[-1] == raw[-1]
    assert [sequence.length(i) for i in range(len(raw))] == [len(r) for r in raw]
    assert list(sequence) == raw


def test_dataset_npz_round_trip(tmp_path):
    src = [[4, 5, 6], [7, 8], [9]]
    tgt = [[10, 11], [12], [13, 14, 15]]
    dataset = ParallelDataset(src, tgt)
    path = tmp_path / "split.npz"
    dataset.save_npz(path)
    back = ParallelDataset.load_npz(path)
    assert len(back) == len(dataset)
    assert [back.src_ids[i] for i in range(len(src))] == src
    assert [back.tgt_ids[i] for i in range(len(tgt))] == tgt


def test_collate_is_identical_after_npz_round_trip(tmp_path):
    """The storage change must not alter a single tensor the model sees."""
    srcVocab, tgtVocab = tiny_vocabs()
    src = [[4, 5, 6], [7, 8], [9, 10, 11, 12]]
    tgt = [[13, 14], [15], [16, 17, 18]]
    fresh = ParallelDataset(src, tgt)
    path = tmp_path / "rt.npz"
    fresh.save_npz(path)
    reloaded = ParallelDataset.load_npz(path)

    for reverse in (False, True):
        a = collate(fresh, [0, 1, 2], srcVocab, tgtVocab, reverseSource=reverse)
        b = collate(reloaded, [0, 1, 2], srcVocab, tgtVocab, reverseSource=reverse)
        for x, y in zip(a, b):
            assert torch.equal(x, y), "npz round-trip changed the collated tensors"
        # and the tensors must still be exactly what the raw lists imply,
        # up to the row's true length (the rest is padding to the batch max)
        expected = (list(reversed(src[0])) if reverse else src[0]) + [srcVocab.eos_id]
        n = int(a[1][0])
        assert n == len(expected)
        assert a[0][0][:n].tolist() == expected
        assert a[0][0][n:].tolist() == [srcVocab.pad_id] * (a[0].size(1) - n)


def test_bucketing_matches_true_lengths():
    """make_batches now sorts on the precomputed length array, not len() calls."""
    src = [[4] * (i % 30 + 1) for i in range(300)]
    dataset = ParallelDataset(src, [[5] * (i % 30 + 1) for i in range(300)])
    assert list(dataset.src_ids.lengths) == [len(s) for s in src]
    batches = make_batches(dataset, 16, shuffle=True, seed=0)
    assert sorted(i for b in batches for i in b) == list(range(300))


# chunked, checkpointed CE loss

def test_chunked_loss_matches_unchunked_value_and_gradients():
    """The memory optimisation must not change the mathematics.

    Same loss and same gradients as computing the full logits tensor at once.
    """
    from seq2seq.losses import chunked_ce_loss

    torch.manual_seed(21)
    model, srcVocab, tgtVocab = tiny_model(seed=21)
    dataset = ParallelDataset([[4, 5, 6, 7], [8, 9], [10, 11, 12]],
                         [[13, 14, 15], [16], [17, 18, 19, 20]])
    src, src_len, tgt_in, tgt_out = collate(dataset, [0, 1, 2], srcVocab, tgtVocab, reverseSource=True)

    def grads_for(chunk):
        model.zero_grad(set_to_none=True)
        hidden = model.forward_hidden(src, src_len, tgt_in)
        loss = chunked_ce_loss(hidden, model.decoder.out, tgt_out, tgtVocab.pad_id, chunk)
        loss.backward()
        g = torch.cat([path.grad.flatten().clone() for path in model.parameters() if path.grad is not None])
        return loss.item(), g

    full, g_full = grads_for(0)
    for chunk in (1, 2, 3, 8):
        chunked, g_chunk = grads_for(chunk)
        assert abs(full - chunked) < 1e-4, f"chunk={chunk}: loss {full} vs {chunked}"
        assert (g_full - g_chunk).abs().max() < 1e-5, f"chunk={chunk}: gradient mismatch"


def test_chunked_loss_matches_plain_cross_entropy():
    """And both agree with torch's own cross_entropy on the full logits."""
    from seq2seq.losses import chunked_ce_loss

    model, srcVocab, tgtVocab = tiny_model(seed=23)
    dataset = ParallelDataset([[4, 5, 6]], [[7, 8, 9]])
    src, src_len, tgt_in, tgt_out = collate(dataset, [0], srcVocab, tgtVocab, reverseSource=False)
    with torch.no_grad():
        logits = model(src, src_len, tgt_in)
        reference = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1),
            ignore_index=tgtVocab.pad_id, reduction="sum")
        hidden = model.forward_hidden(src, src_len, tgt_in)
        got = chunked_ce_loss(hidden, model.decoder.out, tgt_out, tgtVocab.pad_id, 2)
    assert torch.allclose(reference, got, atol=1e-5)
