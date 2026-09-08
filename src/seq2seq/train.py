"""Training loop -- the paper's optimiser, schedule and clipping.

Owner: M2 (model & training).

Held exactly as in Sutskever et al. sec. 3.4:
  * plain SGD, no momentum, lr 0.7;
  * lr constant for the first stretch, then HALVED every half epoch;
  * batches of 128 sequences, and the loss is the summed token NLL DIVIDED BY
    THE BATCH SIZE -- so `clip_grad_norm_(params, 5)` is literally the paper's
    rule "compute s = ||g||_2 where g is the gradient divided by 128; if s > 5
    set g = 5g/s". Dividing by tokens instead would silently change the clip
    threshold's meaning, which is why we do not.
  * length-bucketed minibatches.

Reported perplexity is per-token, exp(sum NLL / target tokens), comparable in
kind to the paper's 5.8 (forward) vs 4.7 (reversed).
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


def lr_at(epoch_float: float, base_lr: float, const_epochs: float, halve_every: float) -> float:
    """Paper: constant, then halved every `halve_every` epochs."""
    if epoch_float < const_epochs:
        return base_lr
    steps = int((epoch_float - const_epochs) / halve_every) + 1
    return base_lr * (0.5**steps)


def load_prepared(data_dir: Path):
    """Load vocabularies and encoded splits.

    Prefers the compact .npz form; if only the JSON exists it converts once and
    caches the .npz beside it. The JSON parses into Python int objects (~1 GB for
    the 500k-pair training set) which the OS then pages out under memory
    pressure; the .npz is int32 and stays resident.
    """
    src_vocab = Vocab.load(data_dir / "vocab.src.json")
    tgt_vocab = Vocab.load(data_dir / "vocab.tgt.json")
    splits = {}
    for split in ("train", "dev", "test"):
        npz, js = data_dir / f"{split}.npz", data_dir / f"{split}.ids.json"
        if npz.exists():
            splits[split] = ParallelDataset.load_npz(npz)
        elif js.exists():
            obj = json.loads(js.read_text())
            ds = ParallelDataset(obj["src"], obj["tgt"])
            del obj
            try:
                ds.save_npz(npz)
            except OSError:
                pass  # cache is an optimisation, not a requirement
            splits[split] = ds
    return src_vocab, tgt_vocab, splits


@torch.no_grad()
def evaluate_perplexity(model, dataset, src_vocab, tgt_vocab, reverse_source, device,
                        batch_size=128, loss_chunk=0):
    was_training = model.training
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id, reduction="sum")
    total_nll, total_tokens = 0.0, 0
    for batch in make_batches(dataset, batch_size, shuffle=False):
        src, src_len, tgt_in, tgt_out = collate(
            dataset, batch, src_vocab, tgt_vocab, reverse_source, device
        )
        if loss_chunk:
            hidden = model.forward_hidden(src, src_len, tgt_in)
            total_nll += chunked_ce_loss(hidden, model.decoder.out, tgt_out,
                                         tgt_vocab.pad_id, loss_chunk).item()
        else:
            logits = model(src, src_len, tgt_in)
            total_nll += criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)).item()
        total_tokens += int((tgt_out != tgt_vocab.pad_id).sum())
    model.train(was_training)  # restore, rather than assuming we came from training
    return math.exp(total_nll / max(total_tokens, 1)), total_nll, total_tokens


def train(cfg: dict) -> Path:
    device = pick_device(cfg.get("device", "auto"))
    set_seed(cfg["seed"])

    data_dir = Path(cfg["data_dir"]).expanduser()
    src_vocab, tgt_vocab, splits = load_prepared(data_dir)
    train_ds, dev_ds = splits["train"], splits["dev"]

    run_dir = Path(cfg["run_dir"]).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    logger = JsonlLogger(run_dir / "log.jsonl")

    model = build_model(cfg, src_vocab, tgt_vocab).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"])  # no momentum
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id, reduction="sum")

    epochs = cfg["epochs"]
    const_epochs = cfg.get("lr_constant_epochs", epochs * 2.0 / 3.0)
    halve_every = cfg.get("lr_halve_every", 0.5)
    batch_size = cfg["batch_size"]
    clip = cfg.get("grad_clip", 5.0)
    reverse = bool(cfg["reverse_source"])
    # Chunked, checkpointed loss: the 32k-vocabulary logits tensor is the memory
    # bottleneck (128 x 51 x 32000 x 4B = 836 MB). 0 disables it.
    loss_chunk = int(cfg.get("loss_chunk_size", 0))
    empty_cache_every = int(cfg.get("empty_cache_every", 200))

    start_epoch, best_ppl = 0, float("inf")
    last_ckpt = run_dir / "last.pt"
    if cfg.get("resume", True) and last_ckpt.exists():
        ck = load_checkpoint(last_ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["meta"]["epoch"] + 1
        best_ppl = ck["meta"].get("best_ppl", float("inf"))
        print(f"[resume] from epoch {start_epoch} (best dev ppl {best_ppl:.3f})")

    print(
        f"[setup] device={device} reverse_source={reverse} loss_chunk={loss_chunk or 'off'} "
        f"params={model.count_parameters()/1e6:.1f}M "
        f"| train={len(train_ds)} dev={len(dev_ds)} | src_vocab={len(src_vocab)} tgt_vocab={len(tgt_vocab)}"
    )

    for epoch in range(start_epoch, epochs):
        batches = make_batches(train_ds, batch_size, shuffle=True, seed=cfg["seed"] * 1000 + epoch)
        running_nll = running_tokens = 0
        t0 = time.time()
        for step, batch in enumerate(batches):
            epoch_float = epoch + step / len(batches)
            lr = lr_at(epoch_float, cfg["lr"], const_epochs, halve_every)
            for group in optimizer.param_groups:
                group["lr"] = lr

            src, src_len, tgt_in, tgt_out = collate(
                train_ds, batch, src_vocab, tgt_vocab, reverse, device
            )
            if loss_chunk:
                hidden = model.forward_hidden(src, src_len, tgt_in)
                nll = chunked_ce_loss(hidden, model.decoder.out, tgt_out,
                                      tgt_vocab.pad_id, loss_chunk)
            else:
                logits = model(src, src_len, tgt_in)
                nll = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss = nll / len(batch)  # paper: gradient divided by the batch size

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            running_nll += nll.item()
            running_tokens += int((tgt_out != tgt_vocab.pad_id).sum())

            # Release cached allocator blocks periodically. Without this the
            # process grows until the OS starts swapping and throughput
            # collapses (observed: 7000 -> under 1000 target words/s).
            if empty_cache_every and step and step % empty_cache_every == 0:
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()

            if step % cfg.get("log_every", 100) == 0:
                elapsed = max(time.time() - t0, 1e-9)
                logger.log(
                    kind="step", epoch=epoch, step=step, total_steps=len(batches),
                    lr=lr, train_ppl=math.exp(running_nll / max(running_tokens, 1)),
                    grad_norm=float(gnorm), words_per_sec=running_tokens / elapsed,
                )
                print(
                    f"  ep{epoch} {step:>5}/{len(batches)} lr={lr:.4f} "
                    f"ppl={math.exp(running_nll / max(running_tokens,1)):8.3f} "
                    f"gnorm={float(gnorm):6.2f} {running_tokens/elapsed:7.0f} tgt-words/s",
                    flush=True,
                )

        train_ppl = math.exp(running_nll / max(running_tokens, 1))
        dev_ppl, _, _ = evaluate_perplexity(
            model, dev_ds, src_vocab, tgt_vocab, reverse, device, batch_size, loss_chunk
        )
        secs = time.time() - t0
        print(f"[epoch {epoch}] train_ppl={train_ppl:.3f} dev_ppl={dev_ppl:.3f} ({secs/60:.1f} min)", flush=True)
        logger.log(kind="epoch", epoch=epoch, train_ppl=train_ppl, dev_ppl=dev_ppl, seconds=secs)

        meta = {"epoch": epoch, "dev_ppl": dev_ppl, "best_ppl": min(best_ppl, dev_ppl), "config": cfg}
        save_checkpoint(last_ckpt, model, optimizer, meta)
        if dev_ppl < best_ppl:
            best_ppl = dev_ppl
            save_checkpoint(run_dir / "best.pt", model, optimizer, meta)
            print(f"  -> new best dev ppl {best_ppl:.3f}", flush=True)

    logger.log(kind="done", best_dev_ppl=best_ppl)
    return run_dir / "best.pt"


def load_config(path: Path, overrides: dict) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg["data_dir"] = str(Path(str(cfg["data_dir"]).replace("$WORK", str(work_dir()))))
    cfg["run_dir"] = str(Path(str(cfg["run_dir"]).replace("$WORK", str(work_dir()))))
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Train the seq2seq LSTM (Sutskever et al. 2014).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--reverse-source", dest="reverse_source", action="store_true", default=None)
    ap.add_argument("--forward-source", dest="reverse_source", action="store_false")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--run-dir", dest="run_dir", default=None)
    ap.add_argument("--data-dir", dest="data_dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, vars(args))
    if args.run_dir is None:
        tag = "rev" if cfg["reverse_source"] else "fwd"
        cfg["run_dir"] = str(Path(cfg["run_dir"]) / f"{tag}_seed{cfg['seed']}")
    train(cfg)


if __name__ == "__main__":
    main()
