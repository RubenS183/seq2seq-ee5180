"""Decode a test set and score it two ways.

Owner: M3 (decoding & eval).

The paper (sec. 3.6) used cased `multi-bleu.pl` on TOKENISED text. sacreBLEU on
detokenised text is the modern reproducible standard. The two are not
interchangeable, so we always report BOTH:

  bleu_tok  -- cased BLEU on Moses-tokenised hyp/ref (`--tokenize none`).
               This is the multi-bleu.pl-equivalent, comparable IN KIND to
               Table 1. It is the headline number.
  bleu_detok -- sacreBLEU on detokenised output vs raw references, with the
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

import torch
from tqdm import tqdm

from .beam import beam_search
from .data import ParallelDataset, detokenize, read_lines
from .model import build_model
from .train import evaluate_perplexity, load_prepared
from .utils import load_checkpoint, pick_device, work_dir
from .vocab import Vocab

LENGTH_BUCKETS = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 10**6)]


def load_models(ckpt_paths, src_vocab, tgt_vocab, device):
    models, metas = [], []
    for path in ckpt_paths:
        ck = load_checkpoint(Path(path), map_location="cpu")
        cfg = ck["meta"]["config"]
        model = build_model(cfg, src_vocab, tgt_vocab)
        model.load_state_dict(ck["model"])
        model.to(device).eval()
        models.append(model)
        metas.append(ck["meta"])
    return models, metas


def score_tokenized(hyps_tok: list[str], refs_tok: list[str]):
    from sacrebleu.metrics import BLEU

    bleu = BLEU(tokenize="none", force=True)
    res = bleu.corpus_score(hyps_tok, [refs_tok])
    return res.score, bleu.get_signature().format()


def score_detokenized(hyps: list[str], refs: list[str]):
    from sacrebleu.metrics import BLEU

    bleu = BLEU()  # default 13a tokenizer, cased
    res = bleu.corpus_score(hyps, [refs])
    return res.score, bleu.get_signature().format()


def bleu_by_length(src_tok, hyps_tok, refs_tok):
    out = []
    for lo, hi in LENGTH_BUCKETS:
        idx = [i for i, s in enumerate(src_tok) if lo <= len(s.split()) <= hi]
        if len(idx) < 5:
            continue
        score, _ = score_tokenized([hyps_tok[i] for i in idx], [refs_tok[i] for i in idx])
        out.append({"bucket": f"{lo}-{'inf' if hi > 10**5 else hi}", "n": len(idx), "bleu_tok": score})
    return out


def run_eval(
    ckpts, data_dir, beam_size, reverse_source, device=None, limit=0,
    length_norm=0.0, tgt_lang="fr", src_lang="en", out_dir=None, tag="",
):
    device = pick_device(device or "auto")
    data_dir = Path(data_dir)
    src_vocab, tgt_vocab, splits = load_prepared(data_dir)
    test = splits["test"]
    models, metas = load_models(ckpts, src_vocab, tgt_vocab, device)

    n = len(test) if not limit else min(limit, len(test))
    src_tok_all = read_lines(data_dir / f"test.tok.{src_lang}")
    ref_tok_all = read_lines(data_dir / f"test.tok.{tgt_lang}")
    ref_raw_all = read_lines(data_dir / f"test.raw.{tgt_lang}")

    hyps_tok, t0 = [], time.time()
    for i in tqdm(range(n), desc=f"decode B={beam_size}{(' ' + tag) if tag else ''}", file=sys.stderr):
        ids, _ = beam_search(
            models, test.src_ids[i], src_vocab, tgt_vocab,
            beam_size=beam_size, reverse_source=reverse_source,
            length_norm=length_norm, device=device,
        )
        hyps_tok.append(" ".join(tgt_vocab.decode(ids)))
    decode_secs = time.time() - t0

    refs_tok, src_tok, refs_raw = ref_tok_all[:n], src_tok_all[:n], ref_raw_all[:n]
    hyps_detok = [detokenize(h.split(), tgt_lang) for h in hyps_tok]

    bleu_tok, sig_tok = score_tokenized(hyps_tok, refs_tok)
    bleu_detok, sig_detok = score_detokenized(hyps_detok, refs_raw)

    ppl = None
    if len(ckpts) == 1:
        ppl, _, _ = evaluate_perplexity(
            models[0], ParallelDataset(test.src_ids[:n], test.tgt_ids[:n]),
            src_vocab, tgt_vocab, reverse_source, device,
        )

    result = {
        "checkpoints": [str(c) for c in ckpts],
        "n_models": len(models),
        "beam_size": beam_size,
        "reverse_source": reverse_source,
        "length_norm": length_norm,
        "n_sentences": n,
        "bleu_tok": round(bleu_tok, 2),
        "bleu_detok": round(bleu_detok, 2),
        "test_ppl": round(ppl, 3) if ppl else None,
        "sig_tok": sig_tok,
        "sig_detok": sig_detok,
        "decode_seconds": round(decode_secs, 1),
        "bleu_by_length": bleu_by_length(src_tok, hyps_tok, refs_tok),
        "dev_ppl_at_checkpoint": [m.get("dev_ppl") for m in metas],
    }

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{tag or 'model'}_beam{beam_size}"
        (out_dir / f"{stem}.hyp.tok.{tgt_lang}").write_text("\n".join(hyps_tok) + "\n", encoding="utf-8")
        (out_dir / f"{stem}.hyp.detok.{tgt_lang}").write_text("\n".join(hyps_detok) + "\n", encoding="utf-8")
        (out_dir / f"{stem}.eval.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description="Decode + score (Table 1 rows).")
    ap.add_argument("--checkpoint", nargs="+", required=True, help="1 = single model, >1 = ensemble")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--beam", type=int, default=12)
    ap.add_argument("--reverse-source", dest="reverse_source", action="store_true", default=False)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--length-norm", type=float, default=0.0)
    # Beam search is latency-bound, not throughput-bound: batches are one
    # sentence x B beams, so MPS kernel-launch overhead dominates. Measured on
    # an M1 Pro at B=12: CPU 34 ms/sentence vs MPS 317 ms/sentence -- 9x faster
    # on CPU. Training stays on MPS; decoding defaults to CPU.
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--src-lang", default="en")
    ap.add_argument("--tgt-lang", default="fr")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    res = run_eval(
        args.checkpoint, str(Path(args.data_dir).expanduser()), args.beam,
        args.reverse_source, args.device, args.limit, args.length_norm,
        args.tgt_lang, args.src_lang, args.out_dir, args.tag,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
