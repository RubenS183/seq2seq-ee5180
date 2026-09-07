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
    v = Vocab(["<pad>", "<unk>", "<sos>", "<eos>"] + [f"w{i}" for i in range(20)])
    return v, v


def tiny_model(seed=0):
    torch.manual_seed(seed)
    sv, tv = tiny_vocabs()
    return Seq2Seq(len(sv), len(tv), emb_dim=16, hidden_dim=16, num_layers=2,
                   src_pad_id=sv.pad_id, tgt_pad_id=tv.pad_id), sv, tv


# ----------------------------------------------------------------- data layer

def test_init_range_is_the_papers():
    m, _, _ = tiny_model()
    w = torch.cat([p.flatten() for p in m.parameters()]).detach()
    assert w.min() >= -0.08 and w.max() <= 0.08


def test_reversal_reverses_source_only_and_keeps_eos_last():
    sv, tv = tiny_vocabs()
    ds = ParallelDataset([[4, 5, 6]], [[7, 8]])
    fwd = collate(ds, [0], sv, tv, reverse_source=False)
    rev = collate(ds, [0], sv, tv, reverse_source=True)
    assert fwd[0][0].tolist() == [4, 5, 6, sv.eos_id]
    assert rev[0][0].tolist() == [6, 5, 4, sv.eos_id]      # words reversed, <eos> still last
    assert fwd[3][0].tolist() == rev[3][0].tolist()        # target untouched


def test_collate_targets_are_shifted_by_one():
    sv, tv = tiny_vocabs()
    ds = ParallelDataset([[4, 5]], [[7, 8, 9]])
    _, _, tgt_in, tgt_out = collate(ds, [0], sv, tv, reverse_source=False)
    assert tgt_in[0].tolist() == [tv.sos_id, 7, 8, 9]
    assert tgt_out[0].tolist() == [7, 8, 9, tv.eos_id]


def test_length_bucketing_groups_similar_lengths():
    ds = ParallelDataset([[4] * (i % 40 + 1) for i in range(512)],
                         [[5] * (i % 40 + 1) for i in range(512)])
    batches = make_batches(ds, 32, shuffle=True, seed=0)
    assert sum(len(b) for b in batches) == 512
    spread = max(max(len(ds.src_ids[i]) for i in b) - min(len(ds.src_ids[i]) for i in b)
                 for b in batches)
    assert spread <= 4, f"buckets too loose (spread {spread})"


def test_clean_pairs_filters_and_reports():
    s = [["a"], [], ["b"] * 60, ["c"], ["c"], ["d"]]
    t = [["x"], ["y"], ["z"], ["w"], ["w"], ["e"] * 5]
    ks, kt, st = clean_pairs(s, t, max_len=50, max_ratio=2.5)
    assert st["empty"] == 1 and st["length"] == 1 and st["dupe"] == 1 and st["ratio"] == 1
    assert len(ks) == len(kt) == 2


# --------------------------------------------------------------------- model

def test_forward_shapes():
    m, sv, tv = tiny_model()
    src = torch.randint(4, 20, (3, 7)); src_len = torch.tensor([7, 5, 6])
    logits = m(src, src_len, torch.randint(4, 20, (3, 9)))
    assert logits.shape == (3, 9, len(tv))


def test_padding_does_not_change_the_sentence_vector():
    """Encoding with extra padding must equal encoding the bare sentence."""
    m, sv, tv = tiny_model()
    m.eval()
    seq = [4, 5, 6]
    with torch.no_grad():
        h1, c1 = m.encode(torch.tensor([seq]), torch.tensor([3]))
        padded = torch.tensor([seq + [sv.pad_id] * 4])
        h2, c2 = m.encode(padded, torch.tensor([3]))
    assert torch.allclose(h1, h2, atol=1e-6) and torch.allclose(c1, c2, atol=1e-6)


def test_model_can_overfit_a_tiny_batch():
    """If the training signal is wired up correctly this drives loss to ~0."""
    torch.manual_seed(0)
    m, sv, tv = tiny_model()
    ds = ParallelDataset([[4, 5, 6], [7, 8]], [[9, 10], [11, 12, 13]])
    src, src_len, tgt_in, tgt_out = collate(ds, [0, 1], sv, tv, reverse_source=True)
    opt = torch.optim.SGD(m.parameters(), lr=1.0)
    crit = torch.nn.CrossEntropyLoss(ignore_index=tv.pad_id, reduction="sum")
    first = last = None
    for i in range(300):
        loss = crit(m(src, src_len, tgt_in).reshape(-1, len(tv)), tgt_out.reshape(-1)) / 2
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        first = loss.item() if i == 0 else first
        last = loss.item()
    assert last < 0.05 * first, f"failed to overfit: {first:.3f} -> {last:.3f}"


# ---------------------------------------------------------------- beam search

def test_beam_one_equals_greedy():
    m, sv, tv = tiny_model(seed=3)
    for src in ([4, 5, 6], [7], [8, 9, 10, 11, 12]):
        b, _ = beam_search(m, src, sv, tv, beam_size=1, reverse_source=True)
        g, _ = greedy_decode(m, src, sv, tv, reverse_source=True)
        assert b == g, f"beam(B=1) != greedy for {src}: {b} vs {g}"


def test_wider_beam_never_finds_a_worse_hypothesis():
    m, sv, tv = tiny_model(seed=5)
    for src in ([4, 5, 6], [8, 9, 10, 11]):
        _, s1 = beam_search(m, src, sv, tv, beam_size=1, reverse_source=True)
        _, s12 = beam_search(m, src, sv, tv, beam_size=12, reverse_source=True)
        assert s12 >= s1 - 1e-5, f"B=12 scored worse than B=1: {s12} < {s1}"


def test_beam_respects_reversal_flag():
    m, sv, tv = tiny_model(seed=7)
    a, _ = beam_search(m, [4, 5, 6, 7], sv, tv, beam_size=4, reverse_source=False)
    b, _ = beam_search(m, [7, 6, 5, 4], sv, tv, beam_size=4, reverse_source=True)
    assert a == b, "reversing the input and setting the flag must cancel out"


# ------------------------------------------------------------------ BLEU gate

def test_bleu_of_references_against_themselves_is_100():
    from seq2seq.evaluate import score_tokenized

    refs = ["le chat est sur le tapis", "un homme marche dans la rue"]
    score, _ = score_tokenized(refs, refs)
    assert score == pytest.approx(100.0, abs=1e-6)


# ----------------------------------------------------------- device agreement

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_mps_matches_cpu_forward_and_backward():
    """Guards against silently wrong Metal LSTM kernels."""
    torch.manual_seed(11)
    m, sv, tv = tiny_model(seed=11)
    ds = ParallelDataset([[4, 5, 6], [7, 8], [9, 10, 11, 12]], [[13, 14], [15], [16, 17, 18]])
    crit = torch.nn.CrossEntropyLoss(ignore_index=tv.pad_id, reduction="sum")

    def loss_and_grad(device):
        mm = Seq2Seq(len(sv), len(tv), emb_dim=16, hidden_dim=16, num_layers=2,
                     src_pad_id=sv.pad_id, tgt_pad_id=tv.pad_id)
        mm.load_state_dict(m.state_dict())
        mm = mm.to(device)
        src, src_len, tgt_in, tgt_out = collate(ds, [0, 1, 2], sv, tv, True, torch.device(device))
        loss = crit(mm(src, src_len, tgt_in).reshape(-1, len(tv)), tgt_out.reshape(-1)) / 3
        loss.backward()
        grad = torch.cat([p.grad.flatten().cpu() for p in mm.parameters() if p.grad is not None])
        return loss.item(), grad

    lc, gc = loss_and_grad("cpu")
    lm, gm = loss_and_grad("mps")
    assert abs(lc - lm) < 1e-3, f"loss mismatch cpu={lc} mps={lm}"
    assert (gc - gm).abs().max() < 1e-3, f"grad mismatch {(gc - gm).abs().max()}"


# ------------------------------------------------- corpus reading (alignment)

def test_read_lines_splits_on_newline_only(tmp_path):
    """Regression: News-Commentary v9 carries ~2900 bare \\r *inside* lines, and a
    different count per language. Python's universal-newline mode split on them
    and silently misaligned the parallel corpus."""
    from seq2seq.data import read_lines

    p = tmp_path / "corpus.txt"
    p.write_bytes(
        "one\rwith carriage return\n"
        "two with unicode separator\n"
        "three\n".encode("utf-8")
    )
    lines = read_lines(p)
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
