"""Encoder / decoder LSTMs -- the paper's architecture, scaled down, plus the
end-term's attention arm behind a flag.

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

END-TERM. `attention="bahdanau"` swaps `Decoder` for `AttnDecoder`, which reads
the encoder's per-position annotations instead of only its final state. That is
the ONLY difference between the two arms we report -- same data, same seed, same
schedule -- so the comparison isolates the bottleneck. Two structural notes:

  * The readout projects [h_t ; c_t] back down to `hidden_dim`, so `self.out`
    keeps shape (hidden_dim -> vocab) and `losses.chunked_ce_loss` -- the thing
    that makes a 32k softmax fit in 16 GB -- needs no change at all.
  * The encoder stays UNIDIRECTIONAL by default. Bahdanau et al. use a BiRNN,
    but adding bidirectionality at the same time as attention would confound
    the two; `encoder_bidirectional` is a separate flag and a declared
    deviation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .attention import BahdanauAttention, source_mask

ATTENTION_KINDS = (None, "none", "bahdanau")


class DecodeState:
    """Everything the decoder carries from one step to the next.

    For the paper's model that is just the LSTM's (h, c). For the attention
    model it also holds the encoder memory bank, its precomputed projection and
    the source mask. Bundling them means `beam.py` and `greedy_decode` keep ONE
    implementation that serves both architectures -- the alternative was two
    divergent beam searches, and `tests/test_smoke.py` asserts that beam search
    at B=1 is exactly greedy decoding, which is a gate two copies would fail.

    `reorder` is the beam-survivor operation. It doubles as beam *expansion*:
    the beam starts one hypothesis wide and `reorder` is called with repeated
    indices, so index_select grows the batch dimension for free.
    """

    __slots__ = ("rnn", "enc_out", "enc_proj", "mask")

    def __init__(self, rnn, enc_out=None, enc_proj=None, mask=None):
        self.rnn = rnn
        self.enc_out = enc_out
        self.enc_proj = enc_proj
        self.mask = mask

    @property
    def attends(self) -> bool:
        return self.enc_out is not None

    def with_rnn(self, rnn) -> "DecodeState":
        """Same memory bank, advanced recurrent state."""
        return DecodeState(rnn, self.enc_out, self.enc_proj, self.mask)

    def reorder(self, index) -> "DecodeState":
        h, c = self.rnn
        rnn = (h[:, index, :], c[:, index, :])
        if not self.attends:
            return DecodeState(rnn)
        return DecodeState(rnn, self.enc_out[index], self.enc_proj[index], self.mask[index])


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, pad_id, dropout=0.0,
                 bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        # A BiLSTM's per-direction states must be merged before they can
        # initialise a unidirectional decoder of the same width.
        self.merge_h = nn.Linear(2 * hidden_dim, hidden_dim) if bidirectional else None
        self.merge_c = nn.Linear(2 * hidden_dim, hidden_dim) if bidirectional else None

    @property
    def out_dim(self) -> int:
        """Width of one annotation h_j -- what attention reads."""
        return self.hidden_dim * (2 if self.bidirectional else 1)

    def _merge(self, state, proj):
        # (layers*2, B, H) -> (layers, B, 2H) -> (layers, B, H)
        layers, batch = self.num_layers, state.size(1)
        state = state.view(layers, 2, batch, self.hidden_dim)
        state = torch.cat([state[:, 0], state[:, 1]], dim=-1)
        return torch.tanh(proj(state))

    def forward(self, src, src_len):
        """Returns (annotations, sentence vector).

        annotations: (batch, src_width, out_dim) -- one vector per source
        position, which the paper's model throws away and attention needs. It is
        padded back out to `src.size(1)` so its indices line up with `src`,
        which keeps the alignment heatmaps and the source mask consistent.
        """
        emb = self.embedding(src)
        packed = pack_padded_sequence(
            emb, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        out, (h, c) = self.rnn(packed)
        enc_out, _ = pad_packed_sequence(
            out, batch_first=True, total_length=src.size(1)
        )
        if self.bidirectional:
            h, c = self._merge(h, self.merge_h), self._merge(c, self.merge_c)
        return enc_out, (h, c)  # the sentence vector: (num_layers, batch, hidden) x 2


class Decoder(nn.Module):
    """The paper's decoder: a language model initialised from the sentence vector."""

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

    def forward(self, tgt_in, state: DecodeState):
        output, state = self.hidden(tgt_in, state)
        return self.out(output), state

    def hidden(self, tgt_in, state: DecodeState):
        """Decoder states BEFORE the output projection.

        Kept separate so the loss can project in chunks: the projected logits
        over a 32k vocabulary are the memory bottleneck, not the LSTM itself
        (see losses.chunked_ce_loss).

        One fused LSTM call over the whole target -- the fast path. Only the
        attention arm pays for a per-step loop.
        """
        emb = self.embedding(tgt_in)
        output, rnn = self.rnn(emb, state.rnn)
        return output, state.with_rnn(rnn)


class AttnDecoder(nn.Module):
    """Bahdanau et al. (2015), additive attention.

    s_t = LSTM([emb(y_{t-1}) ; c_t], s_{t-1}) where c_t is attention over the
    encoder annotations keyed on s_{t-1} -- the state as it stands BEFORE this
    step, which is the ordering in the paper's eq. (4)-(6). At t=0 that query is
    the encoder's final top-layer h, i.e. exactly the baseline's sentence
    vector: attention starts from the bottleneck and is then allowed to look
    back.

    Because c_t enters the recurrence, the timesteps cannot be computed in one
    fused call -- hence the loop. Measured on an M1 Pro: the decoder LSTM alone
    is ~3x slower looped, but it is only ~9% of a training step (the 32k softmax
    dominates), so the end-to-end penalty is ~1.4x.

    Readout. The paper's eq. (10) is a maxout over
        U_o s_i + V_o E y_{i-1} + C_o c_i,
    i.e. three terms. We keep all three -- dropping E y_{i-1} would quietly make
    this Luong's readout rather than Bahdanau's -- but use tanh in place of the
    maxout. That is our one declared readout deviation, and the reason is
    comparability: tanh bounds the readout, so `out` sees inputs of the same
    magnitude as the baseline's raw LSTM output, the initial loss stays at
    ~ln|V|, and Sutskever's lr 0.7 / clip 5.0 schedule transfers to this arm
    untouched. An unbounded maxout readout would have needed its own tuning,
    and then a BLEU difference between the arms could be an optimisation
    artefact instead of an architectural result.
    """

    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, pad_id, dropout=0.0,
                 enc_dim=None, attn_dim=None):
        super().__init__()
        enc_dim = enc_dim or hidden_dim
        attn_dim = attn_dim or hidden_dim
        self.enc_dim = enc_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.attention = BahdanauAttention(enc_dim, hidden_dim, attn_dim)
        self.rnn = nn.LSTM(
            emb_dim + enc_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Projects [h_t ; c_t ; emb(y_{t-1})] back down to hidden_dim, so `out`
        # -- and therefore losses.chunked_ce_loss and its gradient-equality
        # tests -- is unchanged.
        self.readout = nn.Linear(hidden_dim + enc_dim + emb_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_in, state: DecodeState):
        output, state = self.hidden(tgt_in, state)
        return self.out(output), state

    def hidden(self, tgt_in, state: DecodeState, return_weights: bool = False):
        if not state.attends:
            raise ValueError("AttnDecoder needs an encoder memory bank; "
                             "build the state with Seq2Seq.encode_for_decode")
        emb = self.embedding(tgt_in)
        rnn = state.rnn
        outs, weights = [], []
        for t in range(tgt_in.size(1)):
            # Attention is keyed on s_{t-1}: the top layer's h as it stands
            # BEFORE this step consumes y_{t-1}. That ordering is Bahdanau's.
            context, a = self.attention(
                rnn[0][-1], state.enc_out, state.enc_proj, state.mask
            )
            step_in = torch.cat([emb[:, t], context], dim=-1).unsqueeze(1)
            output, rnn = self.rnn(step_in, rnn)
            # U_o s_i + C_o c_i + V_o E y_{i-1}, as one Linear over the
            # concatenation (paper eq. 10, tanh instead of maxout).
            outs.append(torch.tanh(self.readout(
                torch.cat([output.squeeze(1), context, emb[:, t]], dim=-1)
            )))
            if return_weights:
                weights.append(a)
        hidden = torch.stack(outs, dim=1)
        new_state = state.with_rnn(rnn)
        if return_weights:
            return hidden, new_state, torch.stack(weights, dim=1)
        return hidden, new_state


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
        attention=None,
        attn_dim=None,
        encoder_bidirectional=False,
    ):
        super().__init__()
        if attention in (None, "none"):
            attention = None
        if attention not in ATTENTION_KINDS:
            raise ValueError(f"unknown attention {attention!r}; expected one of {ATTENTION_KINDS}")
        if attention is None and encoder_bidirectional:
            raise ValueError("encoder_bidirectional is only supported with attention: "
                             "a BiLSTM's annotations have nothing to read them")
        self.attention_kind = attention
        self.encoder = Encoder(
            src_vocab_size, emb_dim, hidden_dim, num_layers, src_pad_id, dropout,
            bidirectional=encoder_bidirectional,
        )
        if attention == "bahdanau":
            self.decoder = AttnDecoder(
                tgt_vocab_size, emb_dim, hidden_dim, num_layers, tgt_pad_id, dropout,
                enc_dim=self.encoder.out_dim, attn_dim=attn_dim,
            )
        else:
            self.decoder = Decoder(
                tgt_vocab_size, emb_dim, hidden_dim, num_layers, tgt_pad_id, dropout
            )
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.init_range = init_range
        self.reset_parameters()

    def reset_parameters(self):
        """Paper sec. 3.4: uniform between -0.08 and 0.08.

        Applied to the attention parameters too. The paper's rule is stated for
        "all of the LSTM's parameters"; extending it rather than importing a
        modern scheme keeps one declared initialisation across both arms.
        """
        r = self.init_range
        for p in self.parameters():
            nn.init.uniform_(p, -r, r)

    # ------------------------------------------------------------------ encode

    def encode(self, src, src_len):
        """The sentence vector -- the paper's 8000 real numbers.

        Deliberately still returns just (h, c): this is the bottleneck object
        the project is about, and tests assert padding does not change it.
        Decoding goes through `encode_for_decode` instead.
        """
        _, state = self.encoder(src, src_len)
        return state

    def encode_for_decode(self, src, src_len) -> DecodeState:
        """Everything the decoder needs to start: recurrent state, and for the
        attention arm the memory bank, its projection and the source mask."""
        enc_out, rnn = self.encoder(src, src_len)
        if self.attention_kind is None:
            return DecodeState(rnn)
        mask = source_mask(src_len, enc_out.size(1))
        return DecodeState(rnn, enc_out, self.decoder.attention.precompute(enc_out), mask)

    # ------------------------------------------------------------------- train

    def forward(self, src, src_len, tgt_in):
        logits, _ = self.decoder(tgt_in, self.encode_for_decode(src, src_len))
        return logits

    def forward_hidden(self, src, src_len, tgt_in):
        """Decoder hidden states, for the chunked loss path."""
        hidden, _ = self.decoder.hidden(tgt_in, self.encode_for_decode(src, src_len))
        return hidden

    # ------------------------------------------------------------------ decode

    def decode_step(self, token, state: DecodeState):
        """One timestep. The single entry point beam search and greedy use."""
        return self.decoder(token, state)

    @torch.no_grad()
    def alignments(self, src, src_len, tgt_in):
        """Attention weights (batch, tgt_time, src_width) for the heatmaps."""
        if self.attention_kind is None:
            return None
        _, _, weights = self.decoder.hidden(
            tgt_in, self.encode_for_decode(src, src_len), return_weights=True
        )
        return weights

    # -------------------------------------------------------------------- info

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
        # Absent from every mid-term config and checkpoint, so those keep
        # building the paper's model unchanged.
        attention=cfg.get("attention"),
        attn_dim=cfg.get("attn_dim"),
        encoder_bidirectional=cfg.get("encoder_bidirectional", False),
    )
