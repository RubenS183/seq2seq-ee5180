"""Decode a test set and score it two ways.

The paper (sec. 3.6) used cased `multi-bleu.pl` on TOKENISED text. sacreBLEU on
detokenised text is the modern reproducible standard. The two are not
interchangeable, so we always report both:

  bleu_tok   cased BLEU on Moses-tokenised hyp/ref (`--tokenize none`). This is
             the multi-bleu.pl equivalent, comparable IN KIND to Table 1, and it
             is the headline number.
  bleu_detok sacreBLEU on detokenised output against raw references, with the
             full signature recorded for reproducibility.

Also emits per-source-length-bucket BLEU (the Fig. 3 analogue) and sample
translations (the Table 3 analogue).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

from .beam import beam_search
from .data import ParallelDataset, detokenize, read_lines
from .model import build_model
from .train import evaluate_perplexity, load_prepared
from .utils import load_checkpoint, pick_device
from .vocab import Vocab

# The mid-term's buckets, kept exactly so re-rendering mid-term figures is a no-op.
LENGTH_BUCKETS = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 10**6)]

# The end-term raises the training cap from 50 to 100 tokens and the bottleneck
# is meant to bite hardest at the top end, so resolve the tail instead of lumping
# everything over 40 tokens into one bucket.
END_TERM_BUCKETS = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 70), (71, 10**6)]

BUCKET_SETS = {"midterm": LENGTH_BUCKETS, "endterm": END_TERM_BUCKETS}


def load_models(ckptPaths, srcVocab, tgtVocab, device):
    models, metas = [], []
    for path in ckptPaths:
        checkpoint = load_checkpoint(Path(path), mapLocation="cpu")
        model = build_model(checkpoint["meta"]["config"], srcVocab, tgtVocab)
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        models.append(model)
        metas.append(checkpoint["meta"])
    return models, metas


def score_tokenized(hypsTok: list[str], refsTok: list[str]):
    from sacrebleu.metrics import BLEU

    bleu = BLEU(tokenize="none", force=True)
    result = bleu.corpus_score(hypsTok, [refsTok])
    return result.score, bleu.get_signature().format()


def score_detokenized(hyps: list[str], refs: list[str]):
    from sacrebleu.metrics import BLEU

    bleu = BLEU()  # default 13a tokenizer, cased
    result = bleu.corpus_score(hyps, [refs])
    return result.score, bleu.get_signature().format()


def bleu_by_length(srcTok, hypsTok, refsTok, buckets=None):
    rows = []
    for low, high in (buckets or LENGTH_BUCKETS):
        selected = [
            i for i, source in enumerate(srcTok) if low <= len(source.split()) <= high
        ]
        if len(selected) < 5:
            continue
        score, _ = score_tokenized(
            [hypsTok[i] for i in selected], [refsTok[i] for i in selected]
        )
        rows.append({
            "bucket": f"{low}-{'inf' if high > 10**5 else high}",
            "n": len(selected),
            "bleu_tok": score,
        })
    return rows


def run_eval(
    ckpts, dataDir, beamSize, reverseSource, device=None, limit=0,
    lengthNorm=0.0, tgtLang="fr", srcLang="en", outDir=None, tag="",
    buckets=None, maxDecodeLen=100,
):
    device = pick_device(device or "auto")
    dataDir = Path(dataDir)
    srcVocab, tgtVocab, splits = load_prepared(dataDir)
    testSet = splits["test"]
    models, metas = load_models(ckpts, srcVocab, tgtVocab, device)

    count = len(testSet) if not limit else min(limit, len(testSet))
    allSrcTok = read_lines(dataDir / f"test.tok.{srcLang}")
    allRefTok = read_lines(dataDir / f"test.tok.{tgtLang}")
    allRefRaw = read_lines(dataDir / f"test.raw.{tgtLang}")

    hypsTok, decodeStart = [], time.time()
    label = f"decode B={beamSize}{(' ' + tag) if tag else ''}"
    for index in tqdm(range(count), desc=label, file=sys.stderr):
        ids, _ = beam_search(
            models, testSet.src_ids[index], srcVocab, tgtVocab,
            beamSize=beamSize, reverseSource=reverseSource,
            lengthNorm=lengthNorm, device=device, maxDecodeLen=maxDecodeLen,
        )
        hypsTok.append(" ".join(tgtVocab.decode(ids)))
    decodeSeconds = time.time() - decodeStart

    refsTok, srcTok, refsRaw = allRefTok[:count], allSrcTok[:count], allRefRaw[:count]
    hypsDetok = [detokenize(h.split(), tgtLang) for h in hypsTok]

    bleuTok, sigTok = score_tokenized(hypsTok, refsTok)
    bleuDetok, sigDetok = score_detokenized(hypsDetok, refsRaw)

    perplexity = None
    if len(ckpts) == 1:
        perplexity, _, _ = evaluate_perplexity(
            models[0], ParallelDataset(testSet.src_ids[:count], testSet.tgt_ids[:count]),
            srcVocab, tgtVocab, reverseSource, device,
        )

    result = {
        "checkpoints": [str(c) for c in ckpts],
        "n_models": len(models),
        "beam_size": beamSize,
        "reverse_source": reverseSource,
        "length_norm": lengthNorm,
        "max_decode_len": maxDecodeLen,
        "n_sentences": count,
        "bleu_tok": round(bleuTok, 2),
        "bleu_detok": round(bleuDetok, 2),
        "test_ppl": round(perplexity, 3) if perplexity else None,
        "sig_tok": sigTok,
        "sig_detok": sigDetok,
        "decode_seconds": round(decodeSeconds, 1),
        "bleu_by_length": bleu_by_length(srcTok, hypsTok, refsTok, buckets),
        "dev_ppl_at_checkpoint": [m.get("dev_ppl") for m in metas],
        # Which arm produced this row, read back from the checkpoint's own config
        # so a results table can never mislabel a model.
        "attention": metas[0]["config"].get("attention") or "none",
        "n_params": sum(p.numel() for p in models[0].parameters()),
    }

    if outDir:
        outDir = Path(outDir)
        outDir.mkdir(parents=True, exist_ok=True)
        stem = f"{tag or 'model'}_beam{beamSize}"
        (outDir / f"{stem}.hyp.tok.{tgtLang}").write_text(
            "\n".join(hypsTok) + "\n", encoding="utf-8"
        )
        (outDir / f"{stem}.hyp.detok.{tgtLang}").write_text(
            "\n".join(hypsDetok) + "\n", encoding="utf-8"
        )
        (outDir / f"{stem}.eval.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="Decode + score (Table 1 rows).")
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="1 = single model, >1 = ensemble")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--beam", type=int, default=12)
    parser.add_argument("--reverse-source", dest="reverse_source",
                        action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--length-norm", type=float, default=0.0)
    # Beam search is latency-bound, not throughput-bound: batches are one
    # sentence x B beams, so MPS kernel-launch overhead dominates. Measured on an
    # M1 Pro at B=12: CPU 34 ms/sentence against MPS 317 ms/sentence, 9x faster
    # on CPU. Training stays on MPS and decoding defaults to CPU.
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--src-lang", default="en")
    parser.add_argument("--tgt-lang", default="fr")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--buckets", default="midterm", choices=sorted(BUCKET_SETS))
    # 100 is what the mid-term decoded with. The end-term raises it because its
    # training cap is 100 tokens, so a faithful hypothesis can be that long too.
    parser.add_argument("--max-decode-len", dest="max_decode_len", type=int, default=100)
    args = parser.parse_args()

    result = run_eval(
        args.checkpoint, str(Path(args.data_dir).expanduser()), args.beam,
        args.reverse_source, args.device, args.limit, args.length_norm,
        args.tgt_lang, args.src_lang, args.out_dir, args.tag,
        BUCKET_SETS[args.buckets], args.max_decode_len,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
