#!/usr/bin/env python3
"""Explanatory figures for the report and slides.

  1. model_schematic.png  paper Fig. 1 analogue: the encoder reads the source
     REVERSED and its final (h, c) is the whole message passed to the decoder.
  2. time_lag.png         why reversal helps: the minimal time lag between the
     first source word and the first target word collapses from T to 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ENCODER_COLOR, DECODER_COLOR, VECTOR_COLOR = "#4c72b0", "#55a868", "#c44e52"


def _box(ax, x, y, width, height, text, color, fontsize=11, alpha=0.20):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02",
                                linewidth=1.6, edgecolor=color,
                                facecolor=color, alpha=alpha))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=fontsize, color="black")


def _arrow(ax, start, end, color="#444444", style="-|>", lw=1.4):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=13,
                                 linewidth=lw, color=color,
                                 shrinkA=2, shrinkB=2))


def model_schematic(outPath: Path):
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 15.6); ax.set_ylim(0, 5.4); ax.axis("off")

    # What the encoder actually reads. data.py reverses the WORDS and then
    # appends <eos>, so <eos> stays the final encoder input in both arms --
    # the sentence vector is always read off the same event.
    encoderInput = ["C", "B", "A", "<eos>"]
    readingOrder = ["A", "B", "C", "<eos>"]
    targetTokens = ["W", "X", "Y", "Z", "<eos>"]

    cellY, cellH, cellW, gap = 2.5, 0.8, 1.15, 0.28
    x = 0.5
    encoderX = []
    for token in encoderInput:
        _box(ax, x, cellY, cellW, cellH, "LSTM", ENCODER_COLOR)
        ax.text(x + cellW / 2, cellY - 0.55, token, ha="center", fontsize=11,
                weight="bold")
        _arrow(ax, (x + cellW / 2, cellY - 0.30), (x + cellW / 2, cellY))
        encoderX.append(x)
        x += cellW + gap
    for left, right in zip(encoderX, encoderX[1:]):
        _arrow(ax, (left + cellW, cellY + cellH / 2), (right, cellY + cellH / 2),
               ENCODER_COLOR)

    vectorX = x + 0.15
    _box(ax, vectorX, cellY - 0.12, 1.5, cellH + 0.24,
         "$v$\n(h, c)", VECTOR_COLOR, fontsize=11, alpha=0.30)
    _arrow(ax, (encoderX[-1] + cellW, cellY + cellH / 2),
           (vectorX, cellY + cellH / 2), ENCODER_COLOR)

    x = vectorX + 1.5 + gap
    decoderX = []
    for step, token in enumerate(targetTokens):
        _box(ax, x, cellY, cellW, cellH, "LSTM", DECODER_COLOR)
        ax.text(x + cellW / 2, cellY + cellH + 0.42, token, ha="center",
                fontsize=11, weight="bold")
        _arrow(ax, (x + cellW / 2, cellY + cellH), (x + cellW / 2, cellY + cellH + 0.28))
        fedIn = "<sos>" if step == 0 else targetTokens[step - 1]
        ax.text(x + cellW / 2, cellY - 0.55, fedIn, ha="center", fontsize=9,
                color="#666666")
        _arrow(ax, (x + cellW / 2, cellY - 0.30), (x + cellW / 2, cellY),
               color="#999999")
        decoderX.append(x)
        x += cellW + gap
    _arrow(ax, (vectorX + 1.5, cellY + cellH / 2), (decoderX[0], cellY + cellH / 2),
           VECTOR_COLOR, lw=2.0)
    for left, right in zip(decoderX, decoderX[1:]):
        _arrow(ax, (left + cellW, cellY + cellH / 2), (right, cellY + cellH / 2),
               DECODER_COLOR)

    ax.text(encoderX[0], 4.8, "ENCODER  — reads the source REVERSED",
            fontsize=11, weight="bold", color=ENCODER_COLOR)
    ax.text(decoderX[0], 4.8, "DECODER  — a language model conditioned on $v$",
            fontsize=11, weight="bold", color=DECODER_COLOR)
    ax.text(vectorX - 0.1, 1.35,
            "the ENTIRE source sentence,\ncompressed to a fixed vector\n"
            "(paper: 8000 numbers; ours: 2048)",
            fontsize=8.5, color=VECTOR_COLOR, ha="left", va="top")
    ax.text(0.5, 0.55, "source:  " + " ".join(readingOrder[:-1]) +
            "        →  fed to the encoder as:  " + " ".join(encoderInput) +
            "        (words reversed, <eos> last; the target is NEVER reversed)",
            fontsize=9, color="#333333")
    ax.set_title("Sequence-to-sequence with a reversed source (Sutskever et al., 2014, Fig. 1)",
                 fontsize=12.5, pad=14)
    fig.tight_layout()
    outPath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outPath, dpi=170)
    plt.close(fig)
    return outPath


def time_lag(outPath: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    sourceTokens = ["A", "B", "C", "D", "E"]
    targetTokens = ["A'", "B'", "C'", "D'", "E'"]

    for ax, reverse in zip(axes, (False, True)):
        shown = list(reversed(sourceTokens)) if reverse else sourceTokens
        for position, token in enumerate(shown):
            ax.text(position, 1.0, token, ha="center", va="center", fontsize=13,
                    weight="bold", color=ENCODER_COLOR)
        for position, token in enumerate(targetTokens):
            ax.text(position + len(sourceTokens) + 0.6, 1.0, token, ha="center",
                    va="center", fontsize=13, weight="bold", color=DECODER_COLOR)
        ax.axvline(len(sourceTokens) - 0.2, color="#bbbbbb", linestyle=":", linewidth=1)

        positionOfA = shown.index("A")
        lag = len(sourceTokens) - positionOfA  # steps from reading A to emitting A'
        ax.annotate("", xy=(len(sourceTokens) + 0.6, 0.72), xytext=(positionOfA, 0.72),
                    arrowprops=dict(arrowstyle="<->", color="#c44e52", lw=1.8))
        ax.text((positionOfA + len(sourceTokens) + 0.6) / 2, 0.52,
                f"minimal time lag = {lag:.0f}", ha="center", fontsize=10,
                color="#c44e52")
        ax.set_title("reversed source" if reverse else "forward source",
                     fontsize=12, weight="bold")
        ax.set_xlim(-0.8, len(sourceTokens) * 2 + 0.6)
        ax.set_ylim(0.3, 1.35)
        ax.axis("off")

    fig.suptitle("Why reversal helps: A and its translation A' become adjacent "
                 "(paper sec. 3.3)", fontsize=12)
    fig.tight_layout()
    outPath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outPath, dpi=170)
    plt.close(fig)
    return outPath


def objective(outPath: Path):
    """The training objective, rendered with matplotlib mathtext.

    The report pipeline has no MathJax or LaTeX, so display equations become
    images rather than raw dollar-sign soup.
    """
    expression = (r"$\dfrac{1}{|\mathcal{S}|}\sum_{(T,S)\in\mathcal{S}} \log\, p(T \mid S)"
            r"\qquad\mathrm{where}\qquad p(T \mid S) = \prod_{t}\ \ p(y_t \mid y_{<t},\, v)$")
    fig = plt.figure(figsize=(8.4, 1.05))
    fig.text(0.5, 0.5, expression, ha="center", va="center", fontsize=17,
             color="#18204a")
    outPath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outPath, dpi=200, transparent=True, bbox_inches="tight",
                pad_inches=0.18)
    plt.close(fig)
    return outPath


if __name__ == "__main__":
    outDir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/figures")
    print(model_schematic(outDir / "model_schematic.png"))
    print(time_lag(outDir / "time_lag.png"))
    print(objective(outDir / "objective.png"))
