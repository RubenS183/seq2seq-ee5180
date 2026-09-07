"""Device selection, seeding, work-dir resolution, checkpoint I/O.

Owner: M2 (model & training).
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def work_dir() -> Path:
    """Root for heavy artifacts (data, checkpoints).

    Deliberately outside the iCloud-synced course folder: a training run that
    checkpoints into iCloud thrashes sync and risks eviction mid-run.
    """
    return Path(os.environ.get("EE5180_WORK", Path.home() / "ee5180-work")).expanduser()


def repo_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, model, optimizer, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "meta": meta,
        },
        tmp,
    )
    tmp.replace(path)  # atomic: an interrupted save never leaves a corrupt checkpoint


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


class JsonlLogger:
    """Append-only training log; survives interruption, easy to re-plot."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **record) -> None:
        with self.path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open() as fh:
            return [json.loads(line) for line in fh if line.strip()]
