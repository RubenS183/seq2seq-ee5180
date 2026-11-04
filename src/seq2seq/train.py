"""Training loop: the paper's optimiser, schedule and clipping.

Held exactly as in Sutskever et al. sec. 3.4:
  * plain SGD, no momentum, lr 0.7;
  * lr constant for the first stretch, then HALVED every half epoch;
  * batches of 128 sequences, and the loss is the summed token NLL DIVIDED BY
    THE BATCH SIZE, so `clip_grad_norm_(params, 5)` is literally the paper's
    rule "compute s = ||g||_2 where g is the gradient divided by 128; if s > 5
    set g = 5g/s". Dividing by tokens instead would silently change what the
    clip threshold means, which is why we do not;
  * length-bucketed minibatches.

Reported perplexity is per-token, exp(sum NLL / target tokens), comparable in
kind to the paper's 5.8 forward vs 4.7 reversed.

The end-term's attention arm trains through this same loop with the same
optimiser, schedule and clipping, so a difference between the arms cannot be an
optimisation artefact.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from .data import ParallelDataset, collate, make_batches
from .losses import chunked_ce_loss
from .model import build_model
from .utils import (
    JsonlLogger,
    load_checkpoint,
    pick_device,
    save_checkpoint,
    set_seed,
    work_dir,
)
from .vocab import Vocab


def lr_at(epochFloat: float, baseLr: float, constEpochs: float, halveEvery: float) -> float:
    """Paper: constant, then halved every `halveEvery` epochs."""
    if epochFloat < constEpochs:
        return baseLr
    halvings = int((epochFloat - constEpochs) / halveEvery) + 1
    return baseLr * (0.5**halvings)


def load_prepared(dataDir: Path):
    """Load vocabularies and encoded splits.

    Prefers the compact .npz form, and if only the JSON exists it converts once
    and caches the .npz beside it. The JSON parses into Python int objects
    (~1 GB for the 500k-pair training set) which the OS then pages out under
    memory pressure, while the .npz is int32 and stays resident.
    """
    srcVocab = Vocab.load(dataDir / "vocab.src.json")
    tgtVocab = Vocab.load(dataDir / "vocab.tgt.json")
    splits = {}
    for split in ("train", "dev", "test"):
        npzPath, jsonPath = dataDir / f"{split}.npz", dataDir / f"{split}.ids.json"
        if npzPath.exists():
            splits[split] = ParallelDataset.load_npz(npzPath)
        elif jsonPath.exists():
            stored = json.loads(jsonPath.read_text())
            dataset = ParallelDataset(stored["src"], stored["tgt"])
            del stored
            try:
                dataset.save_npz(npzPath)
            except OSError:
                pass  # the cache is an optimisation, not a requirement
            splits[split] = dataset
    return srcVocab, tgtVocab, splits


@torch.no_grad()
def evaluate_perplexity(model, dataset, srcVocab, tgtVocab, reverseSource, device,
                        batchSize=128, lossChunk=0):
    wasTraining = model.training
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=tgtVocab.pad_id, reduction="sum")
    totalNll, totalTokens = 0.0, 0
    for batch in make_batches(dataset, batchSize, shuffle=False):
        src, srcLen, tgtIn, tgtOut = collate(
            dataset, batch, srcVocab, tgtVocab, reverseSource, device
        )
        if lossChunk:
            hidden = model.forward_hidden(src, srcLen, tgtIn)
            totalNll += chunked_ce_loss(hidden, model.decoder.out, tgtOut,
                                        tgtVocab.pad_id, lossChunk).item()
        else:
            logits = model(src, srcLen, tgtIn)
            totalNll += criterion(
                logits.reshape(-1, logits.size(-1)), tgtOut.reshape(-1)
            ).item()
        totalTokens += int((tgtOut != tgtVocab.pad_id).sum())
    model.train(wasTraining)  # restore, rather than assuming we came from training
    return math.exp(totalNll / max(totalTokens, 1)), totalNll, totalTokens


def train(cfg: dict) -> Path:
    device = pick_device(cfg.get("device", "auto"))
    set_seed(cfg["seed"])

    dataDir = Path(cfg["data_dir"]).expanduser()
    srcVocab, tgtVocab, splits = load_prepared(dataDir)
    trainSet, devSet = splits["train"], splits["dev"]

    runDir = Path(cfg["run_dir"]).expanduser()
    runDir.mkdir(parents=True, exist_ok=True)
    (runDir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    logger = JsonlLogger(runDir / "log.jsonl")

    model = build_model(cfg, srcVocab, tgtVocab).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"])  # no momentum
    criterion = nn.CrossEntropyLoss(ignore_index=tgtVocab.pad_id, reduction="sum")

    epochs = cfg["epochs"]
    constEpochs = cfg.get("lr_constant_epochs", epochs * 2.0 / 3.0)
    halveEvery = cfg.get("lr_halve_every", 0.5)
    batchSize = cfg["batch_size"]
    clip = cfg.get("grad_clip", 5.0)
    reverse = bool(cfg["reverse_source"])
    # Chunked, checkpointed loss. The 32k-vocabulary logits tensor is the memory
    # bottleneck at 128 x 51 x 32000 x 4B = 836 MB. 0 disables it.
    lossChunk = int(cfg.get("loss_chunk_size", 0))
    emptyCacheEvery = int(cfg.get("empty_cache_every", 200))

    startEpoch, bestPpl = 0, float("inf")
    lastCkpt = runDir / "last.pt"
    if cfg.get("resume", True) and lastCkpt.exists():
        checkpoint = load_checkpoint(lastCkpt, mapLocation=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        startEpoch = checkpoint["meta"]["epoch"] + 1
        bestPpl = checkpoint["meta"].get("best_ppl", float("inf"))
        print(f"[resume] from epoch {startEpoch} (best dev ppl {bestPpl:.3f})")

    print(
        f"[setup] device={device} reverse_source={reverse} "
        f"attention={cfg.get('attention') or 'none'} loss_chunk={lossChunk or 'off'} "
        f"params={model.count_parameters()/1e6:.1f}M "
        f"| train={len(trainSet)} dev={len(devSet)} "
        f"| src_vocab={len(srcVocab)} tgt_vocab={len(tgtVocab)}"
    )

    for epoch in range(startEpoch, epochs):
        batches = make_batches(trainSet, batchSize, shuffle=True,
                               seed=cfg["seed"] * 1000 + epoch)
        runningNll = runningTokens = 0
        epochStart = time.time()
        for step, batch in enumerate(batches):
            epochFloat = epoch + step / len(batches)
            lr = lr_at(epochFloat, cfg["lr"], constEpochs, halveEvery)
            for group in optimizer.param_groups:
                group["lr"] = lr

            src, srcLen, tgtIn, tgtOut = collate(
                trainSet, batch, srcVocab, tgtVocab, reverse, device
            )
            if lossChunk:
                hidden = model.forward_hidden(src, srcLen, tgtIn)
                nll = chunked_ce_loss(hidden, model.decoder.out, tgtOut,
                                      tgtVocab.pad_id, lossChunk)
            else:
                logits = model(src, srcLen, tgtIn)
                nll = criterion(logits.reshape(-1, logits.size(-1)), tgtOut.reshape(-1))
            loss = nll / len(batch)  # paper: gradient divided by the batch size

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradNorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            runningNll += nll.item()
            runningTokens += int((tgtOut != tgtVocab.pad_id).sum())

            # Release cached allocator blocks periodically. Without this the
            # process grows until the OS starts swapping and throughput
            # collapses (observed: 7000 to under 1000 target words/s).
            if emptyCacheEvery and step and step % emptyCacheEvery == 0:
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()

            if step % cfg.get("log_every", 100) == 0:
                elapsed = max(time.time() - epochStart, 1e-9)
                logger.log(
                    kind="step", epoch=epoch, step=step, total_steps=len(batches),
                    lr=lr, train_ppl=math.exp(runningNll / max(runningTokens, 1)),
                    grad_norm=float(gradNorm), words_per_sec=runningTokens / elapsed,
                )
                print(
                    f"  ep{epoch} {step:>5}/{len(batches)} lr={lr:.4f} "
                    f"ppl={math.exp(runningNll / max(runningTokens,1)):8.3f} "
                    f"gnorm={float(gradNorm):6.2f} {runningTokens/elapsed:7.0f} tgt-words/s",
                    flush=True,
                )

        trainPpl = math.exp(runningNll / max(runningTokens, 1))
        devPpl, _, _ = evaluate_perplexity(
            model, devSet, srcVocab, tgtVocab, reverse, device, batchSize, lossChunk
        )
        seconds = time.time() - epochStart
        print(f"[epoch {epoch}] train_ppl={trainPpl:.3f} dev_ppl={devPpl:.3f} "
              f"({seconds/60:.1f} min)", flush=True)
        logger.log(kind="epoch", epoch=epoch, train_ppl=trainPpl, dev_ppl=devPpl,
                   seconds=seconds)

        meta = {"epoch": epoch, "dev_ppl": devPpl,
                "best_ppl": min(bestPpl, devPpl), "config": cfg}
        save_checkpoint(lastCkpt, model, optimizer, meta)
        if devPpl < bestPpl:
            bestPpl = devPpl
            save_checkpoint(runDir / "best.pt", model, optimizer, meta)
            print(f"  -> new best dev ppl {bestPpl:.3f}", flush=True)

    logger.log(kind="done", best_dev_ppl=bestPpl)
    return runDir / "best.pt"


def load_config(path: Path, overrides: dict) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg["data_dir"] = str(Path(str(cfg["data_dir"]).replace("$WORK", str(work_dir()))))
    cfg["run_dir"] = str(Path(str(cfg["run_dir"]).replace("$WORK", str(work_dir()))))
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Train the seq2seq LSTM (Sutskever et al. 2014)."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--reverse-source", dest="reverse_source",
                        action="store_true", default=None)
    parser.add_argument("--forward-source", dest="reverse_source", action="store_false")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-dir", dest="run_dir", default=None)
    parser.add_argument("--data-dir", dest="data_dir", default=None)
    parser.add_argument("--attention", default=None, choices=["none", "bahdanau"],
                        help="end-term arm; omitted means the config decides")
    args = parser.parse_args()

    cfg = load_config(args.config, vars(args))
    if args.run_dir is None:
        tag = "rev" if cfg["reverse_source"] else "fwd"
        # No suffix without attention, so the mid-term's {dir}_seed{N} layout,
        # which the Makefile and make_results.py both resolve, is untouched.
        arm = "_attn" if cfg.get("attention") not in (None, "none") else ""
        cfg["run_dir"] = str(Path(cfg["run_dir"]) / f"{tag}{arm}_seed{cfg['seed']}")
    train(cfg)


if __name__ == "__main__":
    main()
