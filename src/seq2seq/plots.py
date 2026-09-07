"""Figures: BLEU-vs-source-length (paper Fig. 3) and training curves.

Owner: M4 (analysis & writing).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FWD_C, REV_C = "#c44e52", "#4c72b0"


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
