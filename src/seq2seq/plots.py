"""Figures: BLEU-vs-source-length (paper Fig. 3), training curves, and the
end-term's bottleneck overlay and alignment heatmaps.

Convention across the end-term figures: COLOUR is the source direction (forward
or reversed) and LINESTYLE is the arm (solid for the paper's fixed vector, dashed
for attention). That way one legend reads both dimensions of the 2x2 grid.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FORWARD_COLOR, REVERSED_COLOR = "#c44e52", "#4c72b0"


def _save(fig, outPath: Path):
    fig.tight_layout()
    outPath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outPath, dpi=160)
    plt.close(fig)
    return outPath


def _arm_of(result: dict) -> str:
    """base or attn, tolerant of rows written before the arm field existed."""
    if result.get("arm"):
        return result["arm"]
    return "base" if result.get("attention", "none") in ("none", None) else "attn"


def plot_bleu_by_length(results: list[dict], outPath: Path, title: str):
    """Paper Fig. 3 analogue: does the reversed model hold up on long sentences?"""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    plotted = False
    for result in results:
        buckets = result.get("bleu_by_length") or []
        if not buckets:
            continue
        direction = "reversed" if result["reverse_source"] else "forward"
        ax.plot(
            [b["bucket"] for b in buckets], [b["bleu_tok"] for b in buckets],
            marker="o", linewidth=2,
            color=REVERSED_COLOR if result["reverse_source"] else FORWARD_COLOR,
            label=f"{direction}, beam {result['beam_size']}",
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("source sentence length (tokens)")
    ax.set_ylabel("tokenized cased BLEU")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, outPath)


def plot_training_curves(runDirs: dict[str, Path], outPath: Path, title: str):
    """Dev perplexity per epoch. The reversed curve should sit below the forward one."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    plotted = False
    for label, runDir in runDirs.items():
        logPath = Path(runDir) / "log.jsonl"
        if not logPath.exists():
            continue
        rows = [json.loads(line) for line in logPath.read_text().splitlines() if line.strip()]
        perEpoch = [(r["epoch"], r["dev_ppl"]) for r in rows if r.get("kind") == "epoch"]
        if not perEpoch:
            continue
        if "_attn" in label:
            style = "--"
        else:
            style = "-" if "seed1" in label or "seed" not in label else "--"
        ax.plot(
            [epoch + 1 for epoch, _ in perEpoch], [ppl for _, ppl in perEpoch],
            marker="o", ms=3, linewidth=1.8, linestyle=style, label=label,
            color=REVERSED_COLOR if "rev" in label else FORWARD_COLOR,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("epoch")
    ax.set_ylabel("dev perplexity (per token)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return _save(fig, outPath)


def plot_beam_sweep(results: list[dict], outPath: Path, title: str):
    """BLEU against beam size, the Table 1 trend where beam 2 recovers most of beam 12."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for reversed_, color, label in ((True, REVERSED_COLOR, "reversed"),
                                   (False, FORWARD_COLOR, "forward")):
        points = sorted(
            [(r["beam_size"], r["bleu_tok"]) for r in results
             if r["reverse_source"] == reversed_ and r["n_models"] == 1],
            key=lambda point: point[0],
        )
        if points:
            ax.plot([p[0] for p in points], [p[1] for p in points],
                    marker="o", color=color, label=label, linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 12])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("beam size B")
    ax.set_ylabel("tokenized cased BLEU")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, outPath)


def plot_bleu_by_length_overlay(results: list[dict], outPath: Path, title: str):
    """The end-term's headline figure.

    The bottleneck claim is not "attention scores higher", it is "the fixed
    vector degrades as the source gets longer and attention does not". That is a
    statement about the SHAPE of these curves, so both arms belong on one axis
    with a shared y-scale. A flat attention curve beside a falling baseline is
    the whole argument in one picture.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    plotted = False
    for result in results:
        buckets = result.get("bleu_by_length") or []
        if not buckets:
            continue
        arm = _arm_of(result)
        direction = "reversed" if result["reverse_source"] else "forward"
        ax.plot(
            [b["bucket"] for b in buckets], [b["bleu_tok"] for b in buckets],
            marker="o" if arm == "attn" else "s",
            markersize=5,
            color=REVERSED_COLOR if result["reverse_source"] else FORWARD_COLOR,
            linestyle="--" if arm == "attn" else "-",
            label=f"{direction}, {'attention' if arm == 'attn' else 'fixed vector'}",
            linewidth=2,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("source sentence length (tokens)")
    ax.set_ylabel("tokenized cased BLEU")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    return _save(fig, outPath)


def plot_alignment_heatmap(weights, srcTokens, tgtTokens, outPath: Path, title: str):
    """Bahdanau et al. Fig. 3 analogue: one sentence's alignment matrix.

    `weights` is (tgtSteps, srcPositions) and each ROW sums to 1. The source axis
    is in the order the ENCODER read it, reversed for a reversed model, because
    that is what the weights index. Relabelling it to reading order would
    misrepresent which position the model actually attended to.
    """
    import numpy as np

    matrix = np.asarray(weights)[: len(tgtTokens), : len(srcTokens)]
    fig, ax = plt.subplots(figsize=(max(4.5, 0.46 * len(srcTokens) + 2.2),
                                    max(3.2, 0.40 * len(tgtTokens) + 1.8)))
    image = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0,
                      interpolation="nearest")
    ax.set_xticks(range(len(srcTokens)))
    ax.set_xticklabels(srcTokens, rotation=90, fontsize=8)
    ax.set_yticks(range(len(tgtTokens)))
    ax.set_yticklabels(tgtTokens, fontsize=8)
    ax.set_xlabel("source (encoder reading order)")
    ax.set_ylabel("generated target")
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="attention weight")
    return _save(fig, outPath)


def plot_bottleneck_probe(lengths, norms, outPath: Path, title: str):
    """Does the sentence vector saturate as the source gets longer?

    If a single fixed vector has to carry more as sentences grow, its coordinates
    cannot keep growing without bound, so the per-length mean of ||v|| flattening
    out is direct evidence of the capacity limit, independent of BLEU.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(lengths, norms, marker="o", color=REVERSED_COLOR, linewidth=2)
    ax.set_xlabel("source sentence length (tokens)")
    ax.set_ylabel(r"mean $\|v\|_2$ of the sentence vector")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return _save(fig, outPath)


def plot_arm_bars(rows, outPath: Path, title: str):
    """Side-by-side BLEU per (direction, arm), the 2x2 grid as a bar chart.

    Reads best on Tier 0, where three seeds give real error bars.
    """
    import numpy as np

    positions = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(
        positions,
        [r["mean"] for r in rows],
        yerr=[r.get("std", 0.0) for r in rows],
        capsize=4,
        color=[REVERSED_COLOR if r.get("reverse") else FORWARD_COLOR for r in rows],
        edgecolor="white",
    )
    for bar, row in zip(bars, rows):
        if row.get("arm") == "attn":
            bar.set_hatch("//")
    ax.set_xticks(positions)
    ax.set_xticklabels([r["label"] for r in rows], fontsize=9)
    ax.set_ylabel("tokenized cased BLEU")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, outPath)
