"""Correctness gates for the end-term's attention arm.

Two kinds of gate here. The first kind guards attention itself -- masking,
alignment normalisation, the context wiring. The second guards the refactor that
made attention possible: `beam.py` now drives both architectures through one
`DecodeState`, so the gates that already held for the paper's model have to hold
for the attention model too, or the two arms are not comparable.

The MPS check is repeated for this arm deliberately. Attention adds bmm, softmax
and masked_fill kernels to the Metal path, and the whole reason the mid-term has
such a check is that silently wrong Metal numerics cost a day of training.

There is deliberately NO overfit-a-tiny-batch test here, unlike test_smoke.py.
Two examples under plain SGD put this arm on a plateau it escapes somewhere
between step 600 and 2000 depending on the seed, so the test was measuring a
plateau rather than whether the gradient is wired up. The copy-task gate below
covers the same property far more sharply -- a context carrying no gradient
fails it outright, as the no-attention arm demonstrably does -- and
`test_attention_parameters_receive_gradient` asserts gradient flow directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seq2seq.attention import source_mask  # noqa: E402
from seq2seq.beam import beam_search, greedy_decode  # noqa: E402
from seq2seq.data import ParallelDataset, collate  # noqa: E402
from seq2seq.losses import chunked_ce_loss  # noqa: E402
from seq2seq.model import DecodeState, Seq2Seq, build_model  # noqa: E402
from seq2seq.vocab import Vocab  # noqa: E402


def tiny_vocabs():
    vocab = Vocab(["<pad>", "<unk>", "<sos>", "<eos>"] + [f"w{i}" for i in range(20)])
    return vocab, vocab


def alignment_spread(model, src, src_len, tgt_in) -> float:
    """Largest max-minus-min alignment weight across REAL source positions.

    Masking matters: padding carries exactly 0.0 weight, so an unmasked
    max-minus-min would report ~1/src_len for a perfectly uniform alignment and
    tell us nothing about whether attention is peaked.
    """
    w = model.alignments(src, src_len, tgt_in)
    pad = source_mask(src_len, src.size(1)).unsqueeze(1).expand_as(w)
    hi = w.masked_fill(pad, float("-inf")).amax(-1)
    lo = w.masked_fill(pad, float("inf")).amin(-1)
    return (hi - lo).max().item()


def attn_model(seed=0, **kw):
    torch.manual_seed(seed)
    srcVocab, tgtVocab = tiny_vocabs()
    opts = dict(embDim=16, hiddenDim=16, numLayers=2, attention="bahdanau")
    opts.update(kw)
    return Seq2Seq(len(srcVocab), len(tgtVocab), srcPadId=srcVocab.pad_id, tgtPadId=tgtVocab.pad_id, **opts), srcVocab, tgtVocab


# the mask

def test_source_mask_marks_exactly_the_padding():
    mask = source_mask(torch.tensor([3, 1, 5]), 5)
    assert mask.tolist() == [
        [False, False, False, True, True],
        [False, True, True, True, True],
        [False, False, False, False, False],
    ]


# attention core

def test_alignments_are_a_distribution_and_exactly_zero_on_padding():
    """Probability mass on <pad> would mean attending to nothing at all.

    `collate` rounds the source width up to a multiple of 8, so padding is
    essentially always present -- this is not a corner case.
    """
    model, srcVocab, tgtVocab = attn_model()
    model.eval()
    dataset = ParallelDataset([[4, 5, 6], [7, 8], [9, 10, 11, 12]], [[13, 14], [15], [16, 17, 18]])
    src, src_len, tgt_in, _ = collate(dataset, [0, 1, 2], srcVocab, tgtVocab, reverseSource=True)
    w = model.alignments(src, src_len, tgt_in)

    assert w.shape == (3, tgt_in.size(1), src.size(1))
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6)
    pad = source_mask(src_len, src.size(1)).unsqueeze(1).expand_as(w)
    assert w[pad].abs().max().item() == 0.0, "attention leaked onto padding"


def test_padding_does_not_change_the_context_or_the_output():
    """Encoding a bare sentence and a padded one must give identical logits."""
    model, srcVocab, tgtVocab = attn_model()
    model.eval()
    sequence, tgt = [4, 5, 6], [[13, 14]]
    with torch.no_grad():
        bare = model(torch.tensor([sequence]), torch.tensor([3]), torch.tensor(tgt))
        padded = model(torch.tensor([sequence + [srcVocab.pad_id] * 5]), torch.tensor([3]), torch.tensor(tgt))
    assert torch.allclose(bare, padded, atol=1e-6)


def test_the_sentence_vector_accessor_is_unchanged_by_attention():
    """`encode` is the bottleneck object the project is about; it stays (h, c).

    The mid-term asserts padding does not change it, and that test unpacks a
    2-tuple. Attention must not quietly turn it into something else.
    """
    model, srcVocab, tgtVocab = attn_model()
    model.eval()
    with torch.no_grad():
        h, c = model.encode(torch.tensor([[4, 5, 6]]), torch.tensor([3]))
    assert h.shape == c.shape == (2, 1, 16)


def test_the_context_vector_depends_on_the_source():
    """A context wired to the wrong tensor, or detached, still trains and still
    reports plausible perplexity -- it just ignores the encoder. Compare the
    context itself rather than the logits: at initialisation every parameter is
    uniform(+/-0.08), so activations are ~1e-2 and a logit delta would be ~1e-5.
    """
    model, srcVocab, tgtVocab = attn_model(seed=4)
    model.eval()
    srcLen = torch.tensor([4])

    def context(src):
        state = model.encode_for_decode(torch.tensor([src]), srcLen)
        with torch.no_grad():
            context, _ = model.decoder.attention(
                state.rnn[0][-1], state.enc_out, state.enc_proj, state.mask
            )
        return context

    a, b = context([4, 5, 6, 7]), context([9, 10, 11, 12])
    # Relative to the context's own scale: at initialisation attention is
    # near-uniform, so the context is close to a mean over annotations and the
    # two sources differ by ~1% of it. An ignored source differs by exactly 0.
    relative = ((a - b).abs().max() / a.abs().mean()).item()
    assert relative > 1e-3, f"the context vector ignores the source ({relative:.2e})"


def test_attention_learns_a_diagonal_alignment_on_a_copy_task():
    """The gate that attention does what it is for -- and the bottleneck in miniature.

    Target == source over random tokens, which cannot be memorised: the only way
    to emit position t is to look at source position t. So this separates a
    working attention from one that merely compiles.

    Uses Adam rather than the paper's SGD on purpose. This is a correctness gate,
    not a reported experiment; plain SGD at lr 0.7 needs far more than a few
    hundred steps to get a 64-unit model off chance, which would make the test
    useless as a pre-flight check. `scripts/copy_task.py` runs the same task for
    both arms and draws the figure.
    """
    torch.manual_seed(0)
    srcVocab, tgtVocab = tiny_vocabs()
    length, batch, width = 6, 64, 64
    model = Seq2Seq(len(srcVocab), len(tgtVocab), embDim=width, hiddenDim=width, numLayers=1,
                srcPadId=srcVocab.pad_id, tgtPadId=tgtVocab.pad_id, attention="bahdanau")
    generator = torch.Generator().manual_seed(0)

    def make_batch():
        src = torch.randint(4, len(srcVocab), (batch, length), generator=generator)
        return (torch.cat([src, torch.full((batch, 1), srcVocab.eos_id)], 1),
                torch.full((batch,), length + 1),
                torch.cat([torch.full((batch, 1), tgtVocab.sos_id), src], 1),
                torch.cat([src, torch.full((batch, 1), tgtVocab.eos_id)], 1))

    src, src_len, tgt_in, tgt_out = make_batch()
    assert alignment_spread(model, src, src_len, tgt_in) < 1e-3, (
        "the paper's uniform(+/-0.08) init should leave attention near-uniform at step 0"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    crit = torch.nn.CrossEntropyLoss(ignore_index=tgtVocab.pad_id, reduction="sum")
    for _ in range(400):
        src, src_len, tgt_in, tgt_out = make_batch()
        loss = crit(model(src, src_len, tgt_in).reshape(-1, len(tgtVocab)), tgt_out.reshape(-1)) / batch
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    chance = (length + 1) * torch.tensor(float(len(tgtVocab))).log().item()
    assert loss.item() < 0.2 * chance, f"did not learn to copy: {loss.item():.2f} vs chance {chance:.2f}"
    assert alignment_spread(model, src, src_len, tgt_in) > 0.5, "alignment never became peaked"

    # The alignment should be the identity: target step t attends to source t.
    peak = model.alignments(src, src_len, tgt_in)[0, :length].argmax(-1)
    assert torch.equal(peak, torch.arange(length)), f"alignment is not diagonal: {peak.tolist()}"


# DecodeState

def test_decode_state_reorder_matches_manual_indexing():
    model, srcVocab, tgtVocab = attn_model()
    src = torch.tensor([[4, 5, 6, 0], [7, 8, 0, 0]])
    state = model.encode_for_decode(src, torch.tensor([3, 2]))
    indices = torch.tensor([1, 0, 1])
    r = state.reorder(indices)

    assert torch.equal(r.rnn[0], state.rnn[0][:, indices, :])
    assert torch.equal(r.rnn[1], state.rnn[1][:, indices, :])
    assert torch.equal(r.enc_out, state.enc_out[indices])
    assert torch.equal(r.enc_proj, state.enc_proj[indices])
    assert torch.equal(r.mask, state.mask[indices])


def test_decode_state_reorder_expands_a_single_hypothesis_to_a_beam():
    """How the beam actually widens: index_select with repeated indices."""
    model, srcVocab, tgtVocab = attn_model()
    state = model.encode_for_decode(torch.tensor([[4, 5, 6]]), torch.tensor([3]))
    wide = state.reorder(torch.zeros(12, dtype=torch.long))
    assert wide.rnn[0].shape == (2, 12, 16)
    assert wide.enc_out.shape == (12, 3, 16)
    assert torch.equal(wide.enc_out[7], state.enc_out[0])


def test_plain_model_state_carries_no_memory_bank():
    torch.manual_seed(0)
    srcVocab, tgtVocab = tiny_vocabs()
    model = Seq2Seq(len(srcVocab), len(tgtVocab), embDim=16, hiddenDim=16, numLayers=2,
                srcPadId=srcVocab.pad_id, tgtPadId=tgtVocab.pad_id)
    state = model.encode_for_decode(torch.tensor([[4, 5, 6]]), torch.tensor([3]))
    assert not state.attends and state.enc_out is None
    assert isinstance(state.reorder(torch.tensor([0])), DecodeState)
    assert model.alignments(torch.tensor([[4, 5, 6]]), torch.tensor([3]), torch.tensor([[13]])) is None


# decode parity

def test_attention_beam_one_equals_greedy():
    """The mid-term's sharpest gate, re-asserted for the refactored path."""
    model, srcVocab, tgtVocab = attn_model(seed=3)
    for src in ([4, 5, 6], [7], [8, 9, 10, 11, 12]):
        b, _ = beam_search(model, src, srcVocab, tgtVocab, beamSize=1, reverseSource=True)
        g, _ = greedy_decode(model, src, srcVocab, tgtVocab, reverseSource=True)
        assert b == g, f"beam(B=1) != greedy for {src}: {b} vs {g}"


def test_attention_wider_beam_never_finds_a_worse_hypothesis():
    model, srcVocab, tgtVocab = attn_model(seed=5)
    for src in ([4, 5, 6], [8, 9, 10, 11]):
        _, score1 = beam_search(model, src, srcVocab, tgtVocab, beamSize=1, reverseSource=True)
        _, score12 = beam_search(model, src, srcVocab, tgtVocab, beamSize=12, reverseSource=True)
        assert score12 >= score1 - 1e-5, f"B=12 scored worse than B=1: {score12} < {score1}"


def test_attention_beam_respects_reversal_flag():
    model, srcVocab, tgtVocab = attn_model(seed=7)
    a, _ = beam_search(model, [4, 5, 6, 7], srcVocab, tgtVocab, beamSize=4, reverseSource=False)
    b, _ = beam_search(model, [7, 6, 5, 4], srcVocab, tgtVocab, beamSize=4, reverseSource=True)
    assert a == b, "reversing the input and setting the flag must cancel out"


def test_ensembled_attention_models_decode():
    """The ensemble path keeps one DecodeState per member."""
    modelA, srcVocab, tgtVocab = attn_model(seed=1)
    modelB, _, _ = attn_model(seed=2)
    tokenIds, score = beam_search([modelA, modelB], [4, 5, 6], srcVocab, tgtVocab, beamSize=4, reverseSource=True)
    assert isinstance(tokenIds, list) and score <= 0.0


# loss machinery

@pytest.mark.parametrize("chunk", [1, 2, 3, 8])
def test_chunked_loss_matches_unchunked_for_the_attention_model(chunk):
    """The 32k-softmax memory trick must stay exact on this arm too."""
    model, srcVocab, tgtVocab = attn_model(seed=9)
    dataset = ParallelDataset([[4, 5, 6], [7, 8], [9, 10, 11, 12]], [[13, 14], [15], [16, 17, 18]])
    src, src_len, tgt_in, tgt_out = collate(dataset, [0, 1, 2], srcVocab, tgtVocab, reverseSource=True)

    def run(chunk_size):
        model.zero_grad(set_to_none=True)
        hidden = model.forward_hidden(src, src_len, tgt_in)
        loss = chunked_ce_loss(hidden, model.decoder.out, tgt_out, tgtVocab.pad_id, chunk_size)
        loss.backward()
        grad = torch.cat([path.grad.flatten() for path in model.parameters() if path.grad is not None])
        return loss.item(), grad

    ref_loss, ref_grad = run(0)
    got_loss, got_grad = run(chunk)
    assert abs(ref_loss - got_loss) < 1e-4
    assert (ref_grad - got_grad).abs().max() < 1e-4


def test_attention_parameters_receive_gradient():
    model, srcVocab, tgtVocab = attn_model(seed=0)
    dataset = ParallelDataset([[4, 5, 6]], [[9, 10]])
    src, src_len, tgt_in, tgt_out = collate(dataset, [0], srcVocab, tgtVocab, reverseSource=True)
    crit = torch.nn.CrossEntropyLoss(ignore_index=tgtVocab.pad_id, reduction="sum")
    crit(model(src, src_len, tgt_in).reshape(-1, len(tgtVocab)), tgt_out.reshape(-1)).backward()
    for name in ("attention.W_enc", "attention.W_dec", "attention.v", "readout"):
        path = dict(model.decoder.named_parameters())[f"{name}.weight"]
        assert path.grad is not None and path.grad.abs().sum() > 0, f"{name} got no gradient"


# config plumbing

def test_build_model_defaults_to_the_papers_architecture():
    """Every mid-term config and checkpoint omits `attention`."""
    srcVocab, tgtVocab = tiny_vocabs()
    cfg = dict(emb_dim=16, hidden_dim=16, num_layers=2)
    assert build_model(cfg, srcVocab, tgtVocab).attention_kind is None
    assert build_model({**cfg, "attention": "none"}, srcVocab, tgtVocab).attention_kind is None
    assert build_model({**cfg, "attention": "bahdanau"}, srcVocab, tgtVocab).attention_kind == "bahdanau"


def test_unknown_attention_is_rejected_rather_than_ignored():
    srcVocab, tgtVocab = tiny_vocabs()
    with pytest.raises(ValueError):
        build_model({"emb_dim": 16, "hidden_dim": 16, "num_layers": 2,
                     "attention": "luong"}, srcVocab, tgtVocab)


def test_attention_costs_only_a_few_percent_more_parameters():
    """So the arms' BLEU gap cannot be explained away as extra capacity.

    Measured at the reported configuration -- 32k/32k vocabulary, 2 x 512 -- not
    at the toy vocabulary the other tests use: with 24 words the embedding and
    output projection almost vanish and the attention block looks enormous.
    """
    class V:
        def __init__(self, n):
            self.n, self.pad_id = n, 0

        def __len__(self):
            return self.n

    srcVocab = tgtVocab = V(32000)
    cfg = dict(emb_dim=512, hidden_dim=512, num_layers=2)
    base = build_model(cfg, srcVocab, tgtVocab).count_parameters()
    attn = build_model({**cfg, "attention": "bahdanau"}, srcVocab, tgtVocab).count_parameters()
    assert base == 57_588_992, base            # the mid-term's reported 57.6M
    assert 1.0 < attn / base < 1.05, f"attention changed capacity by {attn / base:.3f}x"


def test_bidirectional_encoder_requires_attention():
    """A BiLSTM sentence vector with no attention to read the annotations is a
    confound, not a feature."""
    srcVocab, tgtVocab = tiny_vocabs()
    with pytest.raises(ValueError):
        build_model({"emb_dim": 16, "hidden_dim": 16, "num_layers": 2,
                     "encoder_bidirectional": True}, srcVocab, tgtVocab)


def test_bidirectional_attention_model_trains_and_decodes():
    model, srcVocab, tgtVocab = attn_model(seed=2, encoderBidirectional=True)
    dataset = ParallelDataset([[4, 5, 6]], [[9, 10]])
    src, src_len, tgt_in, tgt_out = collate(dataset, [0], srcVocab, tgtVocab, reverseSource=True)
    assert model.forward_hidden(src, src_len, tgt_in).shape == (1, tgt_in.size(1), 16)
    tokenIds, _ = beam_search(model, [4, 5, 6], srcVocab, tgtVocab, beamSize=3, reverseSource=True)
    assert isinstance(tokenIds, list)


# device agreement

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_mps_matches_cpu_for_the_attention_model():
    """Attention adds bmm / softmax / masked_fill to the Metal path."""
    model, srcVocab, tgtVocab = attn_model(seed=11)
    dataset = ParallelDataset([[4, 5, 6], [7, 8], [9, 10, 11, 12]], [[13, 14], [15], [16, 17, 18]])
    crit = torch.nn.CrossEntropyLoss(ignore_index=tgtVocab.pad_id, reduction="sum")

    def loss_and_grad(device):
        otherModel, _, _ = attn_model(seed=11)
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


# loop / decode-path consistency

def test_teacher_forced_loop_equals_step_by_step_decoding():
    """The gate for an off-by-one in the s_{t-1} attention query.

    Training runs the whole target through `forward_hidden` in one loop; beam
    search feeds one token at a time, carrying the state. If the query were
    taken from s_t instead of s_{t-1}, both paths would still run, both would
    still train, and they would silently disagree -- training would optimise a
    model that decoding does not implement, and BLEU would be mysteriously bad
    with nothing to point at. Demands 1e-6: it is the same arithmetic.
    """
    model, srcVocab, tgtVocab = attn_model(seed=6)
    model.eval()
    src, src_len = torch.tensor([[4, 5, 6, 7]]), torch.tensor([4])
    tgt_in = torch.tensor([[tgtVocab.sos_id, 13, 14, 15, 16]])

    with torch.no_grad():
        whole = model.forward_hidden(src, src_len, tgt_in)
        state = model.encode_for_decode(src, src_len)
        stepped = []
        for t in range(tgt_in.size(1)):
            step, state = model.decoder.hidden(tgt_in[:, t : t + 1], state)
            stepped.append(step)
        stepped = torch.cat(stepped, dim=1)

    assert whole.shape == stepped.shape
    assert torch.allclose(whole, stepped, atol=1e-6), (
        f"teacher-forced and stepped decoding disagree by "
        f"{(whole - stepped).abs().max().item():.2e}"
    )


def test_encoder_outputs_keep_the_collated_width():
    """Guards a performance bug that looks like a mystery slowdown.

    `pad_packed_sequence` returns max(src_len) columns by default, not the width
    `collate` produced. `collate` rounds the source width up to a multiple of 8
    precisely so the number of distinct tensor shapes per epoch stays small --
    data.py explains that Metal's allocator keeps a block per shape and the
    machine ends up swapping. Attention adds four (B, S, .) tensors, so losing
    the quantisation here would reintroduce that failure.
    """
    model, srcVocab, tgtVocab = attn_model()
    dataset = ParallelDataset([[4, 5, 6], [7, 8]], [[13, 14], [15]])
    src, src_len, tgt_in, _ = collate(dataset, [0, 1], srcVocab, tgtVocab, reverseSource=True)
    assert src.size(1) % 8 == 0 and int(src_len.max()) < src.size(1), "set up a padded batch"

    state = model.encode_for_decode(src, src_len)
    assert state.enc_out.size(1) == src.size(1), (
        "encoder annotations were truncated to max(src_len); "
        "pass total_length=src.size(1) to pad_packed_sequence"
    )
    assert state.enc_proj.size(1) == src.size(1)
    # The dropped columns must be genuinely empty, not stale values.
    assert state.enc_out[0, int(src_len[0]):].abs().max().item() == 0.0


def test_a_single_token_source_does_not_produce_nan():
    """A fully-masked softmax row is the only way -inf masking NaNs.

    src_len >= 1 always holds -- collate appends <eos> -- but assert it rather
    than rely on it, in forward and in backward.
    """
    model, srcVocab, tgtVocab = attn_model()
    src = torch.tensor([[srcVocab.eos_id, srcVocab.pad_id, srcVocab.pad_id]])
    src_len = torch.tensor([1])
    tgt_in = torch.tensor([[tgtVocab.sos_id, 13]])

    w = model.alignments(src, src_len, tgt_in)
    assert torch.isfinite(w).all()
    assert torch.allclose(w[:, :, 0], torch.ones_like(w[:, :, 0]))

    logits = model(src, src_len, tgt_in)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    for name, path in model.named_parameters():
        if path.grad is not None:
            assert torch.isfinite(path.grad).all(), f"non-finite gradient in {name}"


# checkpoint round-trip

def test_attention_checkpoint_round_trips_through_its_config(tmp_path):
    """`evaluate.load_models` rebuilds from ck["meta"]["config"], so the arm has
    to be recoverable from the checkpoint alone."""
    from seq2seq.utils import load_checkpoint, save_checkpoint

    srcVocab, tgtVocab = tiny_vocabs()
    cfg = dict(emb_dim=16, hidden_dim=16, num_layers=2, attention="bahdanau")
    torch.manual_seed(5)
    model = build_model(cfg, srcVocab, tgtVocab)
    path = tmp_path / "best.pt"
    save_checkpoint(path, model, None, {"epoch": 0, "config": cfg})

    checkpoint = load_checkpoint(path)
    rebuilt = build_model(checkpoint["meta"]["config"], srcVocab, tgtVocab)
    missing, unexpected = rebuilt.load_state_dict(checkpoint["model"], strict=True)
    assert not missing and not unexpected
    assert rebuilt.attention_kind == "bahdanau"

    src, src_len = torch.tensor([[4, 5, 6]]), torch.tensor([3])
    tgt_in = torch.tensor([[tgtVocab.sos_id, 13]])
    model.eval(); rebuilt.eval()
    with torch.no_grad():
        assert torch.allclose(model(src, src_len, tgt_in), rebuilt(src, src_len, tgt_in))


def test_a_baseline_checkpoint_refuses_to_load_into_an_attention_model():
    """A mis-tagged run must fail loudly, not train on from partial weights."""
    srcVocab, tgtVocab = tiny_vocabs()
    cfg = dict(emb_dim=16, hidden_dim=16, num_layers=2)
    base = build_model(cfg, srcVocab, tgtVocab)
    attn = build_model({**cfg, "attention": "bahdanau"}, srcVocab, tgtVocab)
    with pytest.raises(RuntimeError):
        attn.load_state_dict(base.state_dict(), strict=True)
