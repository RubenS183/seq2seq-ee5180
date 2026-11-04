#!/usr/bin/env python3
"""Alignment heatmaps for a trained attention model (Bahdanau Fig. 3 analogue).

The length curves argue the bottleneck statistically. This shows it directly: for
each generated target token, which source positions did the model read?

Two things to keep straight, both easy to get wrong in a way that would
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
    parser = argparse.ArgumentParser(
        description="Attention heatmaps for a few test sentences."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default="results/alignments")
    parser.add_argument("--beam", type=int, default=12)
    parser.add_argument("--reverse-source", dest="reverse_source",
                        action="store_true", default=False)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-decode-len", dest="max_decode_len", type=int, default=150)
    parser.add_argument("--n", type=int, default=6, help="how many sentences to draw")
    parser.add_argument("--min-len", dest="min_len", type=int, default=6)
    parser.add_argument("--max-len", dest="max_len", type=int, default=22,
                        help="cap so the heatmap stays readable")
    args = parser.parse_args()

    device = pick_device(args.device)
    srcVocab, tgtVocab, splits = load_prepared(Path(args.data_dir).expanduser())
    testSet = splits["test"]
    models, _ = load_models([args.checkpoint], srcVocab, tgtVocab, device)
    model = models[0]
    if model.attention_kind is None:
        sys.exit("this checkpoint has no attention; there are no alignments to plot")

    outDir = Path(args.out)
    outDir.mkdir(parents=True, exist_ok=True)

    # Readable sentences only: a 60-token heatmap is unreadable on a slide.
    picked = [i for i in range(len(testSet))
              if args.min_len <= len(testSet.src_ids[i]) <= args.max_len][: args.n]
    if not picked:
        sys.exit(f"no test sentences with {args.min_len}-{args.max_len} tokens")

    index = []
    for rank, testIndex in enumerate(picked):
        srcIds = list(testSet.src_ids[testIndex])
        hypIds, _ = beam_search(
            model, srcIds, srcVocab, tgtVocab, beamSize=args.beam,
            reverseSource=args.reverse_source, device=device,
            maxDecodeLen=args.max_decode_len,
        )
        if not hypIds:
            continue

        # Re-run the decoder over its own output to recover the weights. The
        # encoder input must be built exactly as collate and beam_search build
        # it: reverse the words, then append <eos>.
        encIds = (list(reversed(srcIds)) if args.reverse_source else list(srcIds))
        encIds = encIds + [srcVocab.eos_id]
        src = torch.tensor([encIds], dtype=torch.long, device=device)
        srcLen = torch.tensor([len(encIds)], dtype=torch.long, device=device)
        # Starts with <sos>, so row t is the step that EMITTED tgtTokens[t].
        tgtIn = torch.tensor([[tgtVocab.sos_id] + hypIds], dtype=torch.long,
                             device=device)

        weights = model.alignments(src, srcLen, tgtIn)[0].cpu().numpy()
        srcTokens = srcVocab.decode(encIds[:-1]) + ["<eos>"]
        tgtTokens = tgtVocab.decode(hypIds)
        direction = "reversed" if args.reverse_source else "forward"
        figurePath = outDir / f"alignment_{rank:02d}.png"
        plot_alignment_heatmap(
            weights[: len(tgtTokens)], srcTokens, tgtTokens, figurePath,
            f"Alignment, {direction} source (test sentence {testIndex})",
        )
        index.append({"test_index": testIndex, "figure": figurePath.name,
                      "source": " ".join(srcVocab.decode(srcIds)),
                      "hypothesis": " ".join(tgtTokens)})
        print(f"  [{rank}] test#{testIndex}: {len(srcIds)} src tokens -> "
              f"{len(tgtTokens)} tgt  {figurePath.name}")

    (outDir / "alignments.json").write_text(json.dumps(index, indent=2))
    markdown = [
        "# Attention alignments (Bahdanau Fig. 3 analogue)", "",
        f"Model: `{args.checkpoint}`, beam {args.beam}, "
        f"{'reversed' if args.reverse_source else 'forward'} source.", "",
        "The source axis is in **encoder reading order**. For a reversed model that is",
        "right-to-left, so a faithful word-by-word alignment shows up as an anti-diagonal.",
        "Alignments are over the model's own output, not the reference.", "",
    ]
    for entry in index:
        markdown += [f"**Source**  {entry['source']}", "",
                     f"**Ours**    {entry['hypothesis']}", "",
                     f"![alignment]({entry['figure']})", "", "---", ""]
    (outDir / "alignments.md").write_text("\n".join(markdown))
    print(f"[done] -> {outDir}")


if __name__ == "__main__":
    main()
