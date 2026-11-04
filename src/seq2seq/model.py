"""Encoder and decoder LSTMs: the paper's architecture scaled down, plus the
end-term's attention arm behind a flag.

Faithful to Sutskever et al. (2014):
  * TWO SEPARATE LSTMs with no shared parameters (sec. 3.4).
  * The encoder's final (h, c) across all layers IS the sentence vector, and the
    decoder is a language model initialised from it and conditioned on nothing
    else. That single fixed-length vector is the bottleneck the end-term studies.
  * Uniform init U(-0.08, 0.08) on every parameter (sec. 3.4).
  * Naive full softmax over the target vocabulary (sec. 3.4).
  * No dropout, so `dropout` defaults to 0.0 and any non-zero value is a
    declared deviation.

Scaled down to 2 layers x 512 cells with 512-d embeddings instead of 4 x 1000.

`attention="bahdanau"` swaps `Decoder` for `AttnDecoder`, which reads the
encoder's per-position annotations instead of only its final state. That is the
ONLY difference between the two arms we report, so the comparison isolates the
bottleneck.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .attention import BahdanauAttention, source_mask

ATTENTION_KINDS = (None, "none", "bahdanau")


class DecodeState:
    """Everything the decoder carries from one step to the next.

    For the paper's model that is just the LSTM's (h, c). For the attention model
    it also holds the encoder memory bank, its precomputed projection and the
    source mask. Bundling them lets `beam.py` keep ONE implementation that serves
    both architectures, and two divergent beam searches would be two places for
    beam(B=1) to stop being greedy decoding.

    `reorder` is the beam-survivor operation. It doubles as beam expansion: the
    beam starts one hypothesis wide and `reorder` is called with repeated
    indices, so index_select grows the batch dimension for free.
    """

    __slots__ = ("rnn", "enc_out", "enc_proj", "mask")

    def __init__(self, rnn, encOut=None, encProj=None, mask=None):
        self.rnn = rnn
        self.enc_out = encOut
        self.enc_proj = encProj
        self.mask = mask

    @property
    def attends(self) -> bool:
        return self.enc_out is not None

    def with_rnn(self, rnn) -> "DecodeState":
        """Same memory bank, advanced recurrent state."""
        return DecodeState(rnn, self.enc_out, self.enc_proj, self.mask)

    def reorder(self, index) -> "DecodeState":
        hidden, cell = self.rnn
        rnn = (hidden[:, index, :], cell[:, index, :])
        if not self.attends:
            return DecodeState(rnn)
        return DecodeState(rnn, self.enc_out[index], self.enc_proj[index], self.mask[index])


class Encoder(nn.Module):
    def __init__(self, vocabSize, embDim, hiddenDim, numLayers, padId, dropout=0.0,
                 bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocabSize, embDim, padding_idx=padId)
        self.bidirectional = bidirectional
        self.hidden_dim = hiddenDim
        self.num_layers = numLayers
        self.rnn = nn.LSTM(
            embDim,
            hiddenDim,
            num_layers=numLayers,
            batch_first=True,
            dropout=dropout if numLayers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        # A BiLSTM's per-direction states must be merged before they can
        # initialise a unidirectional decoder of the same width.
        self.merge_h = nn.Linear(2 * hiddenDim, hiddenDim) if bidirectional else None
        self.merge_c = nn.Linear(2 * hiddenDim, hiddenDim) if bidirectional else None

    @property
    def out_dim(self) -> int:
        """Width of one annotation h_j, which is what attention reads."""
        return self.hidden_dim * (2 if self.bidirectional else 1)

    def _merge(self, state, projection):
        # (layers*2, B, H) -> (layers, B, 2H) -> (layers, B, H)
        layers, batchSize = self.num_layers, state.size(1)
        state = state.view(layers, 2, batchSize, self.hidden_dim)
        state = torch.cat([state[:, 0], state[:, 1]], dim=-1)
        return torch.tanh(projection(state))

    def forward(self, src, srcLen):
        """Returns (annotations, sentence vector).

        The annotations are (batch, srcWidth, outDim), one vector per source
        position, which the paper's model throws away and attention needs. They
        are padded back out to `src.size(1)` so their indices line up with
        `src`, which keeps the alignment heatmaps and the source mask consistent.
        """
        embedded = self.embedding(src)
        packed = pack_padded_sequence(
            embedded, srcLen.cpu(), batch_first=True, enforce_sorted=False
        )
        packedOut, (hidden, cell) = self.rnn(packed)
        encOut, _ = pad_packed_sequence(
            packedOut, batch_first=True, total_length=src.size(1)
        )
        if self.bidirectional:
            hidden = self._merge(hidden, self.merge_h)
            cell = self._merge(cell, self.merge_c)
        return encOut, (hidden, cell)


class Decoder(nn.Module):
    """The paper's decoder: a language model initialised from the sentence vector."""

    def __init__(self, vocabSize, embDim, hiddenDim, numLayers, padId, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocabSize, embDim, padding_idx=padId)
        self.rnn = nn.LSTM(
            embDim,
            hiddenDim,
            num_layers=numLayers,
            batch_first=True,
            dropout=dropout if numLayers > 1 else 0.0,
        )
        self.out = nn.Linear(hiddenDim, vocabSize)

    def forward(self, tgtIn, state: DecodeState):
        output, state = self.hidden(tgtIn, state)
        return self.out(output), state

    def hidden(self, tgtIn, state: DecodeState):
        """Decoder states BEFORE the output projection.

        Kept separate so the loss can project in chunks: the logits over a 32k
        vocabulary are the memory bottleneck, not the LSTM itself. One fused LSTM
        call over the whole target is the fast path, and only the attention arm
        pays for a per-step loop.
        """
        embedded = self.embedding(tgtIn)
        output, rnn = self.rnn(embedded, state.rnn)
        return output, state.with_rnn(rnn)


class AttnDecoder(nn.Module):
    """Bahdanau et al. (2015), additive attention.

    s_t = LSTM([emb(y_{t-1}) ; c_t], s_{t-1}) where c_t is attention over the
    encoder annotations keyed on s_{t-1}, the state as it stands BEFORE this step,
    which is the ordering in the paper's eq. (4)-(6). At t=0 that query is the
    encoder's final top-layer h, i.e. exactly the baseline's sentence vector, so
    attention starts from the bottleneck and is then allowed to look back.

    Because c_t enters the recurrence the timesteps cannot be computed in one
    fused call, hence the loop. Measured on an M1 Pro the decoder LSTM alone is
    ~3x slower looped, but it is only ~9% of a training step because the 32k
    softmax dominates, so the end-to-end penalty is ~1.4x.

    The paper's eq. (10) readout is a maxout over U_o s_i + V_o E y_{i-1} + C_o c_i.
    We keep all three terms, since dropping E y_{i-1} would quietly make this
    Luong's readout rather than Bahdanau's, but use tanh in place of the maxout.
    That is our one declared readout deviation and the reason is comparability:
    tanh bounds the readout, so `out` sees the same magnitudes as the baseline,
    the initial loss stays at ~ln|V|, and Sutskever's lr 0.7 / clip 5.0 schedule
    transfers to this arm untouched. An unbounded maxout readout would have
    needed its own tuning, and then a BLEU difference between the arms could be
    an optimisation artefact instead of an architectural result.
    """

    def __init__(self, vocabSize, embDim, hiddenDim, numLayers, padId, dropout=0.0,
                 encDim=None, attnDim=None):
        super().__init__()
        encDim = encDim or hiddenDim
        attnDim = attnDim or hiddenDim
        self.enc_dim = encDim
        self.embedding = nn.Embedding(vocabSize, embDim, padding_idx=padId)
        self.attention = BahdanauAttention(encDim, hiddenDim, attnDim)
        self.rnn = nn.LSTM(
            embDim + encDim,
            hiddenDim,
            num_layers=numLayers,
            batch_first=True,
            dropout=dropout if numLayers > 1 else 0.0,
        )
        # Projects [h_t ; c_t ; emb(y_{t-1})] back down to hiddenDim, so `out`,
        # and therefore losses.chunked_ce_loss and its gradient-equality tests,
        # is unchanged.
        self.readout = nn.Linear(hiddenDim + encDim + embDim, hiddenDim)
        self.out = nn.Linear(hiddenDim, vocabSize)

    def forward(self, tgtIn, state: DecodeState):
        output, state = self.hidden(tgtIn, state)
        return self.out(output), state

    def hidden(self, tgtIn, state: DecodeState, returnWeights: bool = False):
        if not state.attends:
            raise ValueError("AttnDecoder needs an encoder memory bank; "
                             "build the state with Seq2Seq.encode_for_decode")
        embedded = self.embedding(tgtIn)
        rnn = state.rnn
        outputs, alignments = [], []
        for step in range(tgtIn.size(1)):
            # Attention is keyed on s_{t-1}: the top layer's h as it stands
            # BEFORE this step consumes y_{t-1}. That ordering is Bahdanau's.
            context, weights = self.attention(
                rnn[0][-1], state.enc_out, state.enc_proj, state.mask
            )
            stepInput = torch.cat([embedded[:, step], context], dim=-1).unsqueeze(1)
            output, rnn = self.rnn(stepInput, rnn)
            outputs.append(torch.tanh(self.readout(
                torch.cat([output.squeeze(1), context, embedded[:, step]], dim=-1)
            )))
            if returnWeights:
                alignments.append(weights)
        hidden = torch.stack(outputs, dim=1)
        nextState = state.with_rnn(rnn)
        if returnWeights:
            return hidden, nextState, torch.stack(alignments, dim=1)
        return hidden, nextState


class Seq2Seq(nn.Module):
    def __init__(
        self,
        srcVocabSize,
        tgtVocabSize,
        embDim=512,
        hiddenDim=512,
        numLayers=2,
        srcPadId=0,
        tgtPadId=0,
        dropout=0.0,
        initRange=0.08,
        attention=None,
        attnDim=None,
        encoderBidirectional=False,
    ):
        super().__init__()
        if attention in (None, "none"):
            attention = None
        if attention not in ATTENTION_KINDS:
            raise ValueError(f"unknown attention {attention!r}; expected one of {ATTENTION_KINDS}")
        if attention is None and encoderBidirectional:
            raise ValueError("encoder_bidirectional is only supported with attention: "
                             "a BiLSTM's annotations have nothing to read them")
        self.attention_kind = attention
        self.encoder = Encoder(
            srcVocabSize, embDim, hiddenDim, numLayers, srcPadId, dropout,
            bidirectional=encoderBidirectional,
        )
        if attention == "bahdanau":
            self.decoder = AttnDecoder(
                tgtVocabSize, embDim, hiddenDim, numLayers, tgtPadId, dropout,
                encDim=self.encoder.out_dim, attnDim=attnDim,
            )
        else:
            self.decoder = Decoder(
                tgtVocabSize, embDim, hiddenDim, numLayers, tgtPadId, dropout
            )
        self.num_layers = numLayers
        self.hidden_dim = hiddenDim
        self.init_range = initRange
        self.reset_parameters()

    def reset_parameters(self):
        """Paper sec. 3.4: uniform between -0.08 and 0.08.

        Applied to the attention parameters too. The paper's rule is stated for
        "all of the LSTM's parameters", and extending it rather than importing a
        modern scheme keeps one declared initialisation across both arms.
        """
        bound = self.init_range
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def encode(self, src, srcLen):
        """The sentence vector, the paper's 8000 real numbers.

        Deliberately still returns just (h, c), because this is the bottleneck
        object the project is about and tests assert padding does not change it.
        Decoding goes through `encode_for_decode` instead.
        """
        _, state = self.encoder(src, srcLen)
        return state

    def encode_for_decode(self, src, srcLen) -> DecodeState:
        """Everything the decoder needs to start: recurrent state, and for the
        attention arm the memory bank, its projection and the source mask."""
        encOut, rnn = self.encoder(src, srcLen)
        if self.attention_kind is None:
            return DecodeState(rnn)
        mask = source_mask(srcLen, encOut.size(1))
        return DecodeState(rnn, encOut, self.decoder.attention.precompute(encOut), mask)

    def forward(self, src, srcLen, tgtIn):
        logits, _ = self.decoder(tgtIn, self.encode_for_decode(src, srcLen))
        return logits

    def forward_hidden(self, src, srcLen, tgtIn):
        """Decoder hidden states, for the chunked loss path."""
        hidden, _ = self.decoder.hidden(tgtIn, self.encode_for_decode(src, srcLen))
        return hidden

    def decode_step(self, token, state: DecodeState):
        """One timestep. The single entry point beam search and greedy use."""
        return self.decoder(token, state)

    @torch.no_grad()
    def alignments(self, src, srcLen, tgtIn):
        """Attention weights (batch, tgtTime, srcWidth) for the heatmaps."""
        if self.attention_kind is None:
            return None
        _, _, weights = self.decoder.hidden(
            tgtIn, self.encode_for_decode(src, srcLen), returnWeights=True
        )
        return weights

    @property
    def sentence_vector_size(self) -> int:
        """The paper's '8000 real numbers': numLayers x hidden x (h, c)."""
        return self.num_layers * self.hidden_dim * 2

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg, srcVocab, tgtVocab) -> Seq2Seq:
    """Build from a config dict. The keys stay snake_case because they are read
    straight out of the YAML configs and the checkpoints' saved metadata."""
    return Seq2Seq(
        srcVocabSize=len(srcVocab),
        tgtVocabSize=len(tgtVocab),
        embDim=cfg["emb_dim"],
        hiddenDim=cfg["hidden_dim"],
        numLayers=cfg["num_layers"],
        srcPadId=srcVocab.pad_id,
        tgtPadId=tgtVocab.pad_id,
        dropout=cfg.get("dropout", 0.0),
        initRange=cfg.get("init_range", 0.08),
        # Absent from every mid-term config and checkpoint, so those keep
        # building the paper's model unchanged.
        attention=cfg.get("attention"),
        attnDim=cfg.get("attn_dim"),
        encoderBidirectional=cfg.get("encoder_bidirectional", False),
    )
