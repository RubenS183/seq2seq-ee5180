"""Figures: BLEU-vs-source-length (paper Fig. 3), training curves, and the
end-term's bottleneck overlay and alignment heatmaps.

Owner: M4 (analysis & writing).

Convention across the end-term figures: COLOUR is the source direction
(forward / reversed) and LINESTYLE is the arm (solid = the paper's fixed vector,
dashed = attention). That way one legend reads both dimensions of the 2x2 grid.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FWD_C, REV_C = "#c44e52", "#4c72b0"


def _arm_of(res: dict) -> str:
    """base | attn, tolerant of rows written before the arm field existed."""
    if res.get("arm"):
        return res["arm"]
    return "base" if res.get("attention", "none") in ("none", None) else "attn"


def plot_bleu_by_length(results: list[dict], out_path: Path, title: str):
    """Paper Fig. 3 analogue: does the reversed model hold up on long sentences?"""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    plotted = False
    for res in results:
        buckets = res.get("bleu_by_length") or []
        if not buckets:
            continue
        label = ("reversed" if res["reverse_source"] else "forward") + f", beam {res['beam_size']}"
        color = REV_C if res["reverse_source"] else FWD_C
        ax.plot([b["bucket"] for b in buckets], [b["bleu_tok"] for b in buckets],
                marker="o", color=color, label=label, linewidth=2)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("source sentence length (tokens)")
    ax.set_ylabel("tokenized cased BLEU")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_training_curves(run_dirs: dict[str, Path], out_path: Path, title: str):
    """Dev perplexity per epoch. The reversed curve should sit below the forward one."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    any_data = False
    for label, run_dir in run_dirs.items():
        log = Path(run_dir) / "log.jsonl"
        if not log.exists():
            continue
        rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        epochs = [(r["epoch"], r["dev_ppl"]) for r in rows if r.get("kind") == "epoch"]
        if not epochs:
            continue
        color = REV_C if "rev" in label else FWD_C
        if "_attn" in label:
            style = "--"           # attention arm
        else:
            style = "-" if "seed1" in label or "seed" not in label else "--"
        ax.plot([e + 1 for e, _ in epochs], [p for _, p in epochs],
                marker="o", ms=3, color=color, linestyle=style, label=label, linewidth=1.8)
        any_data = True
    if not any_data:
        plt.close(fig)
        return None
    ax.set_xlabel("epoch")
    ax.set_ylabel("dev perplexity (per token)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_beam_sweep(results: list[dict], out_path: Path, title: str):
    """BLEU vs beam size -- the Table 1 trend that beam 2 recovers most of beam 12."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for rev, color, label in ((True, REV_C, "reversed"), (False, FWD_C, "forward")):
        pts = sorted(
            [(r["beam_size"], r["bleu_tok"]) for r in results
             if r["reverse_source"] == rev and r["n_models"] == 1],
            key=lambda x: x[0],
        )
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    marker="o", color=color, label=label, linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 12])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("beam size B")
    ax.set_ylabel("tokenized cased BLEU")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_bleu_by_length_overlay(results: list[dict], out_path: Path, title: str):
    """The end-term's headline figure.

    The bottleneck claim is not "attention scores higher" -- it is "the fixed
    vector degrades as the source gets longer, and attention does not". That is
    a statement about the SHAPE of these curves, so both arms belong on one axis
    with a shared y-scale. A flat attention curve beside a falling baseline is
    the whole argument in one picture.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    plotted = False
    for res in results:
        buckets = res.get("bleu_by_length") or []
        if not buckets:
            continue
        arm = _arm_of(res)
        direction = "reversed" if res["reverse_source"] else "forward"
        ax.plot(
            [b["bucket"] for b in buckets], [b["bleu_tok"] for b in buckets],
            marker="o" if arm == "attn" else "s",
            markersize=5,
            color=REV_C if res["reverse_source"] else FWD_C,
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
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_alignment_heatmap(weights, src_tokens, tgt_tokens, out_path: Path, title: str):
    """Bahdanau et al. Fig. 3 analogue: one sentence's alignment matrix.

    `weights` is (tgt_steps, src_positions) and each ROW sums to 1. Note the
    source axis is in the order the ENCODER read it -- reversed, for a reversed
    model -- because that is what the weights index. Relabelling it to reading
    order would misrepresent which position the model actually attended to.
    """
    import numpy as np

    w = np.asarray(weights)[: len(tgt_tokens), : len(src_tokens)]
    fig, ax = plt.subplots(figsize=(max(4.5, 0.46 * len(src_tokens) + 2.2),
                                    max(3.2, 0.40 * len(tgt_tokens) + 1.8)))
    im = ax.imshow(w, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0,
                   interpolation="nearest")
    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=90, fontsize=8)
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens, fontsize=8)
    ax.set_xlabel("source (encoder reading order)")
    ax.set_ylabel("generated target")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="attention weight")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_bottleneck_probe(lengths, norms, out_path: Path, title: str):
    """Does the sentence vector saturate as the source gets longer?

    Promised in the mid-term report's section 9. If a single fixed vector has to
    carry more as sentences grow, its coordinates cannot keep growing without
    bound -- so the per-length mean of ||v|| flattening out is direct evidence
    of the capacity limit, independent of BLEU.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(lengths, norms, marker="o", color=REV_C, linewidth=2)
    ax.set_xlabel("source sentence length (tokens)")
    ax.set_ylabel(r"mean $\|v\|_2$ of the sentence vector")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_arm_bars(rows, out_path: Path, title: str):
    """Side-by-side BLEU per (direction, arm) -- the 2x2 grid as a bar chart.

    Reads best on Tier 0, where three seeds give real error bars.
    """
    import numpy as np

    labels = [r["label"] for r in rows]
    means = [r["mean"] for r in rows]
    errs = [r.get("std", 0.0) for r in rows]
    colors = [REV_C if r.get("reverse") else FWD_C for r in rows]
    hatch = ["//" if r.get("arm") == "attn" else None for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(np.arange(len(rows)), means, yerr=errs, capsize=4,
                  color=colors, edgecolor="white")
    for bar, h in zip(bars, hatch):
        if h:
            bar.set_hatch(h)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("tokenized cased BLEU")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
