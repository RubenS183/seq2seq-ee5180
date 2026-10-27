#!/usr/bin/env python3
"""Alignment heatmaps for a trained attention model (Bahdanau Fig. 3 analogue).

Owner: M4 (analysis & writing).

The length curves argue the bottleneck statistically. This shows it directly:
for each generated target token, which source positions did the model read?

Two things to keep straight, both of which are easy to get wrong and would
misrepresent the model:

  * The weights are indexed in the order the ENCODER read the source. For a
    reversed model that is right-to-left, so a correct alignment appears as an
    ANTI-diagonal. We label the axis in encoder order and say so, rather than
    quietly un-reversing it to make a prettier picture.
  * The alignment is computed over the model's OWN output, not the reference.
    Running it against the reference would show how it would have aligned a
    translation it did not produce.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seq2seq.beam import beam_search  # noqa: E402
from seq2seq.evaluate import load_models  # noqa: E402
from seq2seq.plots import plot_alignment_heatmap  # noqa: E402
from seq2seq.train import load_prepared  # noqa: E402
from seq2seq.utils import pick_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Attention heatmaps for a few test sentences.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="results/alignments")
    ap.add_argument("--beam", type=int, default=12)
    ap.add_argument("--reverse-source", dest="reverse_source", action="store_true", default=False)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-decode-len", dest="max_decode_len", type=int, default=150)
    ap.add_argument("--n", type=int, default=6, help="how many sentences to draw")
    ap.add_argument("--min-len", dest="min_len", type=int, default=6)
    ap.add_argument("--max-len", dest="max_len", type=int, default=22,
                    help="cap so the heatmap stays readable")
    args = ap.parse_args()

    device = pick_device(args.device)
    data_dir = Path(args.data_dir).expanduser()
    src_vocab, tgt_vocab, splits = load_prepared(data_dir)
    test = splits["test"]
    models, metas = load_models([args.checkpoint], src_vocab, tgt_vocab, device)
    model = models[0]
    if model.attention_kind is None:
        sys.exit("this checkpoint has no attention; there are no alignments to plot")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Readable sentences only: a 60-token heatmap is unreadable on a slide.
    picked = [i for i in range(len(test))
              if args.min_len <= len(test.src_ids[i]) <= args.max_len][: args.n]
    if not picked:
        sys.exit(f"no test sentences with {args.min_len}-{args.max_len} tokens")

    index = []
    for rank, i in enumerate(picked):
        src_ids = list(test.src_ids[i])
        hyp_ids, _ = beam_search(
            model, src_ids, src_vocab, tgt_vocab, beam_size=args.beam,
            reverse_source=args.reverse_source, device=device,
            max_decode_len=args.max_decode_len,
        )
        if not hyp_ids:
            continue

        # Re-run the decoder over its own output to recover the weights. The
        # encoder input must be built exactly as collate/beam_search build it:
        # reverse the words, then append <eos>.
        enc_ids = list(reversed(src_ids)) if args.reverse_source else list(src_ids)
        enc_ids = enc_ids + [src_vocab.eos_id]
        src = torch.tensor([enc_ids], dtype=torch.long, device=device)
        src_len = torch.tensor([len(enc_ids)], dtype=torch.long, device=device)
        tgt_in = torch.tensor([[tgt_vocab.sos_id] + hyp_ids], dtype=torch.long, device=device)

        weights = model.alignments(src, src_len, tgt_in)[0].cpu().numpy()
        src_tokens = src_vocab.decode(enc_ids[:-1]) + ["<eos>"]
        tgt_tokens = tgt_vocab.decode(hyp_ids)
        # tgt_in starts with <sos>, so row t is the step that EMITTED tgt_tokens[t].
        label = "reversed" if args.reverse_source else "forward"
        path = out / f"alignment_{rank:02d}.png"
        plot_alignment_heatmap(
            weights[: len(tgt_tokens)], src_tokens, tgt_tokens, path,
            f"Alignment, {label} source (test sentence {i})",
        )
        index.append({"test_index": i, "figure": path.name,
                      "source": " ".join(src_vocab.decode(src_ids)),
                      "hypothesis": " ".join(tgt_tokens)})
        print(f"  [{rank}] test#{i}: {len(src_ids)} src tokens -> {len(tgt_tokens)} tgt  {path.name}")

    (out / "alignments.json").write_text(json.dumps(index, indent=2))
    md = ["# Attention alignments (Bahdanau Fig. 3 analogue)", "",
          f"Model: `{args.checkpoint}`, beam {args.beam}, "
          f"{'reversed' if args.reverse_source else 'forward'} source.", "",
          "The source axis is in **encoder reading order**. For a reversed model that is",
          "right-to-left, so a faithful word-by-word alignment shows up as an anti-diagonal.",
          "Alignments are over the model's own output, not the reference.", ""]
    for e in index:
        md += [f"**Source**  {e['source']}", "", f"**Ours**    {e['hypothesis']}", "",
               f"![alignment]({e['figure']})", "", "---", ""]
    (out / "alignments.md").write_text("\n".join(md))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
