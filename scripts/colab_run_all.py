#!/usr/bin/env python3
"""One-shot, resumable driver for the whole WMT'14 pipeline on Colab.

Run it, and it does every remaining stage in order. Run it AGAIN after a
disconnect and it skips whatever already finished and carries on. That is the
whole point: free Colab kills idle runtimes, so recovery has to be one command,
not a dozen clicks.

    python scripts/colab_run_all.py                 # do everything outstanding
    python scripts/colab_run_all.py --status        # just report what is done
    python scripts/colab_run_all.py --dry-run       # show the plan, run nothing
    python scripts/colab_run_all.py --only train_rev decode
    python scripts/colab_run_all.py --seeds 1 2 3 --ensemble   # stretch rows

Stages, in order:
    data      fetch WMT'14, subsample Europarl, tokenise, build vocab -> .npz
    bench     measure GPU throughput (informational)
    train_rev single REVERSED LSTM   (Table 1 row 4)
    train_fwd single FORWARD  LSTM   (Table 1 row 3)
    decode    beam 1/2/12 for both arms, scored two ways, plots
    report    regenerate the mid-term report markdown
    archive   copy results/ + report/ to the work dir on Drive

Notes that cost real time to rediscover:
  * shell scripts are invoked via `bash`, never `./` -- Google Drive's FUSE
    mount strips the executable bit;
  * decoding never overlaps training -- that contention is what crippled an
    earlier attempt;
  * training resumes from last.pt on its own, so an interrupted arm continues
    from its last completed epoch rather than restarting.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STAGES = ["data", "bench", "train_rev", "train_fwd", "decode", "report", "archive"]


def work_dir() -> Path:
    return Path(os.environ.get("EE5180_WORK", Path.home() / "ee5180-work")).expanduser()


def say(msg: str, rule: bool = False) -> None:
    stamp = time.strftime("%H:%M:%S")
    if rule:
        print(f"\n{'=' * 72}\n[{stamp}] {msg}\n{'=' * 72}", flush=True)
    else:
        print(f"[{stamp}] {msg}", flush=True)


def run(cmd: list[str], dry: bool = False) -> None:
    printable = " ".join(str(c) for c in cmd)
    if dry:
        print(f"    would run: {printable}")
        return
    say(f"$ {printable}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO)
    if proc.returncode != 0:
        raise SystemExit(f"\nFAILED (exit {proc.returncode}): {printable}\n"
                         f"Fix the cause, then re-run this script -- finished stages are skipped.")
    say(f"  done in {(time.time() - t0) / 60:.1f} min")


# ------------------------------------------------------------------ state

def data_ready(w: Path) -> bool:
    p = w / "data/wmt14/prepared"
    return all((p / f"{s}.npz").exists() for s in ("train", "dev", "test")) and \
           (p / "vocab.src.json").exists()


def arm_done(w: Path, tag: str) -> bool:
    log = w / "runs/wmt14" / tag / "log.jsonl"
    if not log.exists():
        return False
    return any('"kind": "done"' in line for line in log.read_text().splitlines())


def arm_progress(w: Path, tag: str) -> str:
    log = w / "runs/wmt14" / tag / "log.jsonl"
    if not log.exists():
        return "not started"
    epochs = [json.loads(l) for l in log.read_text().splitlines()
              if l.strip() and '"kind": "epoch"' in l]
    if not epochs:
        return "started, no epoch finished yet"
    last = epochs[-1]
    state = "COMPLETE" if arm_done(w, tag) else "in progress"
    return f"{state}, {len(epochs)} epoch(s), dev ppl {last['dev_ppl']:.2f}"


def decode_done() -> bool:
    return (REPO / "results/wmt14_small/all_results.json").exists()


def status(w: Path) -> None:
    print(f"\nwork dir : {w}")
    print(f"repo     : {REPO}")
    print(f"{'data prepared':<22} {'yes' if data_ready(w) else 'NO'}")
    for tag, label in (("rev_seed1", "reversed arm"), ("fwd_seed1", "forward arm")):
        print(f"{label:<22} {arm_progress(w, tag)}")
    print(f"{'decoded + scored':<22} {'yes' if decode_done() else 'NO'}")
    res = REPO / "results/wmt14_small/results.md"
    if res.exists():
        print("\n--- results/wmt14_small/results.md ---")
        print(res.read_text())


# ----------------------------------------------------------------- stages

SCALE_NOTE = ("0.5M training pairs vs the paper's 12M; 2x512 vs 4x1000; 32k/32k vocab vs "
              "160k/80k. Absolute BLEU is NOT comparable to Table 1 - the direction and "
              "shape of the effects are.")


def stage_data(w: Path, args) -> None:
    if data_ready(w) and not args.force:
        say("data already prepared - skipping")
        return
    raw = w / "data/wmt14/raw"
    run(["bash", "scripts/get_wmt14.sh"], args.dry_run)   # bash, not ./ - Drive strips +x
    if not (raw / "europarl.sub.en").exists() or args.force:
        run([sys.executable, "scripts/subsample_parallel.py",
             "--src-in", str(raw / "europarl.en"), "--tgt-in", str(raw / "europarl.fr"),
             "--src-out", str(raw / "europarl.sub.en"), "--tgt-out", str(raw / "europarl.sub.fr"),
             "--n", str(args.europarl_pairs), "--seed", "1"], args.dry_run)
    run([sys.executable, "scripts/prepare_data.py",
         "--out", str(w / "data/wmt14/prepared"),
         "--train-src", str(raw / "nc9.en"), str(raw / "europarl.sub.en"),
         "--train-tgt", str(raw / "nc9.fr"), str(raw / "europarl.sub.fr"),
         "--dev-src", str(raw / "newstest2013.en"), "--dev-tgt", str(raw / "newstest2013.fr"),
         "--test-src", str(raw / "newstest2014.en"), "--test-tgt", str(raw / "newstest2014.fr"),
         "--src-vocab-size", "32000", "--tgt-vocab-size", "32000",
         "--max-train-pairs", str(args.train_pairs), "--workers", str(args.workers),
         "--seed", "1"], args.dry_run)


def stage_bench(w: Path, args) -> None:
    run([sys.executable, "scripts/benchmark.py", "--corpus-pairs", str(args.train_pairs)],
        args.dry_run)


def _train(w: Path, args, reverse: bool, seed: int) -> None:
    tag = f"{'rev' if reverse else 'fwd'}_seed{seed}"
    if arm_done(w, tag) and not args.force:
        say(f"{tag} already complete ({arm_progress(w, tag)}) - skipping")
        return
    if (w / "runs/wmt14" / tag / "last.pt").exists():
        say(f"{tag}: resuming ({arm_progress(w, tag)})")
    run([sys.executable, "-m", "seq2seq.train", "--config", "configs/wmt14_small.yaml",
         "--reverse-source" if reverse else "--forward-source", "--seed", str(seed)],
        args.dry_run)


def stage_train_rev(w: Path, args) -> None:
    for s in args.seeds:
        _train(w, args, True, s)


def stage_train_fwd(w: Path, args) -> None:
    for s in args.seeds:
        _train(w, args, False, s)


def stage_decode(w: Path, args) -> None:
    if decode_done() and not args.force:
        say("results already present - skipping (use --force to redo)")
        return
    cmd = [sys.executable, "scripts/make_results.py", "--name", "wmt14_small",
           "--data-dir", str(w / "data/wmt14/prepared"),
           "--runs-root", str(w / "runs/wmt14"),
           "--out", "results/wmt14_small", "--beams", "1", "2", "12",
           "--seeds", *[str(s) for s in args.seeds],
           "--device", args.device, "--scale-note", SCALE_NOTE]
    if args.ensemble and len(args.seeds) > 1:
        cmd.append("--ensemble")
    run(cmd, args.dry_run)


def stage_report(w: Path, args) -> None:
    run([sys.executable, "scripts/make_report.py"], args.dry_run)


def stage_archive(w: Path, args) -> None:
    dest = w / "deliverables"
    if args.dry_run:
        print(f"    would copy results/ and report/ -> {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    import shutil
    for name in ("results", "report"):
        src = REPO / name
        if src.exists():
            tgt = dest / name
            if tgt.exists():
                shutil.rmtree(tgt)
            shutil.copytree(src, tgt)
    say(f"copied results/ and report/ -> {dest}")
    say("These live on Drive, so they survive a runtime disconnect.")


RUNNERS = {"data": stage_data, "bench": stage_bench, "train_rev": stage_train_rev,
           "train_fwd": stage_train_fwd, "decode": stage_decode,
           "report": stage_report, "archive": stage_archive}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=STAGES, help="run just these stages")
    ap.add_argument("--skip", nargs="+", choices=STAGES, default=["bench"],
                    help="stages to skip (default: bench)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--ensemble", action="store_true", help="also decode a seed ensemble")
    ap.add_argument("--train-pairs", type=int, default=500000)
    ap.add_argument("--europarl-pairs", type=int, default=1100000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="redo stages even if complete")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true", help="report progress and exit")
    args = ap.parse_args()

    w = work_dir()
    if args.status:
        status(w)
        return

    plan = args.only if args.only else [s for s in STAGES if s not in args.skip]
    say(f"work dir: {w}", rule=True)
    print(f"repo    : {REPO}")
    print(f"plan    : {' -> '.join(plan)}")
    print(f"seeds   : {args.seeds}\n")
    status(w)

    t0 = time.time()
    for name in plan:
        say(f"STAGE: {name}", rule=True)
        RUNNERS[name](w, args)

    say(f"ALL DONE in {(time.time() - t0) / 3600:.2f} h", rule=True)
    status(w)


if __name__ == "__main__":
    main()
