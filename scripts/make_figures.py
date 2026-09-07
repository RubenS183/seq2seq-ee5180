#!/usr/bin/env python3
"""Explanatory figures for the report and slides.

Owner: M4. Two figures:
  1. model_schematic.png  -- paper Fig. 1 analogue: encoder reads the source
     REVERSED, its final (h,c) is the whole message passed to the decoder.
  2. time_lag.png         -- why reversal helps: the minimal time lag between
     the first source word and the first target word collapses from T to 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ENC_C, DEC_C, VEC_C = "#4c72b0", "#55a868", "#c44e52"


def _box(ax, x, y, w, h, text, color, fontsize=11, alpha=0.20):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                linewidth=1.6, edgecolor=color,
                                facecolor=color, alpha=alpha))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="black")


def _arrow(ax, p, q, color="#444444", style="-|>", lw=1.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13,
                                 linewidth=lw, color=color,
                                 shrinkA=2, shrinkB=2))


def model_schematic(out: Path):
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 15.6); ax.set_ylim(0, 5.4); ax.axis("off")

    # What the encoder actually reads. data.py reverses the WORDS and then
    # appends <eos>, so <eos> stays the final encoder input in both arms --
    # the sentence vector is always read off the same event.
    src_display = ["C", "B", "A", "<eos>"]
    src_original = ["A", "B", "C", "<eos>"]
    tgt = ["W", "X", "Y", "Z", "<eos>"]

    y_cell, h, w, gap = 2.5, 0.8, 1.15, 0.28
    x = 0.5
    enc_x = []
    for tok in src_display:
        _box(ax, x, y_cell, w, h, "LSTM", ENC_C)
        ax.text(x + w / 2, y_cell - 0.55, tok, ha="center", fontsize=11, weight="bold")
        _arrow(ax, (x + w / 2, y_cell - 0.30), (x + w / 2, y_cell))
        enc_x.append(x)
        x += w + gap
    for a, b in zip(enc_x, enc_x[1:]):
        _arrow(ax, (a + w, y_cell + h / 2), (b, y_cell + h / 2), ENC_C)

    vec_x = x + 0.15
    _box(ax, vec_x, y_cell - 0.12, 1.5, h + 0.24,
         "$v$\n(h, c)", VEC_C, fontsize=11, alpha=0.30)
    _arrow(ax, (enc_x[-1] + w, y_cell + h / 2), (vec_x, y_cell + h / 2), ENC_C)

    x = vec_x + 1.5 + gap
    dec_x = []
    for i, tok in enumerate(tgt):
        _box(ax, x, y_cell, w, h, "LSTM", DEC_C)
        ax.text(x + w / 2, y_cell + h + 0.42, tok, ha="center", fontsize=11, weight="bold")
        _arrow(ax, (x + w / 2, y_cell + h), (x + w / 2, y_cell + h + 0.28))
        below = "<sos>" if i == 0 else tgt[i - 1]
        ax.text(x + w / 2, y_cell - 0.55, below, ha="center", fontsize=9, color="#666666")
        _arrow(ax, (x + w / 2, y_cell - 0.30), (x + w / 2, y_cell), color="#999999")
        dec_x.append(x)
        x += w + gap
    _arrow(ax, (vec_x + 1.5, y_cell + h / 2), (dec_x[0], y_cell + h / 2), VEC_C, lw=2.0)
    for a, b in zip(dec_x, dec_x[1:]):
        _arrow(ax, (a + w, y_cell + h / 2), (b, y_cell + h / 2), DEC_C)

    ax.text(enc_x[0], 4.8, "ENCODER  — reads the source REVERSED",
            fontsize=11, weight="bold", color=ENC_C)
    ax.text(dec_x[0], 4.8, "DECODER  — a language model conditioned on $v$",
            fontsize=11, weight="bold", color=DEC_C)
    ax.text(vec_x - 0.1, 1.35,
            "the ENTIRE source sentence,\ncompressed to a fixed vector\n"
            "(paper: 8000 numbers; ours: 2048)",
            fontsize=8.5, color=VEC_C, ha="left", va="top")
    ax.text(0.5, 0.55, "source:  " + " ".join(src_original[:-1]) +
            "        →  fed to the encoder as:  " + " ".join(src_display) +
            "        (words reversed, <eos> last; the target is NEVER reversed)",
            fontsize=9, color="#333333")
    ax.set_title("Sequence-to-sequence with a reversed source (Sutskever et al., 2014, Fig. 1)",
                 fontsize=12.5, pad=14)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def time_lag(out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    src = ["A", "B", "C", "D", "E"]
    tgt = ["A'", "B'", "C'", "D'", "E'"]

    for ax, reverse in zip(axes, (False, True)):
        shown = list(reversed(src)) if reverse else src
        for i, tok in enumerate(shown):
            ax.text(i, 1.0, tok, ha="center", va="center", fontsize=13, weight="bold", color=ENC_C)
        for i, tok in enumerate(tgt):
            ax.text(i + len(src) + 0.6, 1.0, tok, ha="center", va="center",
                    fontsize=13, weight="bold", color=DEC_C)
        ax.axvline(len(src) - 0.2, color="#bbbbbb", linestyle=":", linewidth=1)

        pos_a = shown.index("A")
        lag = len(src) - pos_a  # steps from reading A to emitting A'
        ax.annotate("", xy=(len(src) + 0.6, 0.72), xytext=(pos_a, 0.72),
                    arrowprops=dict(arrowstyle="<->", color="#c44e52", lw=1.8))
        ax.text((pos_a + len(src) + 0.6) / 2, 0.52,
                f"minimal time lag = {lag:.0f}", ha="center", fontsize=10, color="#c44e52")
        ax.set_title("reversed source" if reverse else "forward source",
                     fontsize=12, weight="bold")
        ax.set_xlim(-0.8, len(src) * 2 + 0.6); ax.set_ylim(0.3, 1.35); ax.axis("off")

    fig.suptitle("Why reversal helps: A and its translation A' become adjacent (paper sec. 3.3)",
                 fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


if __name__ == "__main__":
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/figures")
    print(model_schematic(out_dir / "model_schematic.png"))
    print(time_lag(out_dir / "time_lag.png"))
