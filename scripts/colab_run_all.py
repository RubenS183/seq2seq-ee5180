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
    printable = " ".join(str(part) for part in cmd)
    if dry:
        print(f"    would run: {printable}")
        return
    say(f"$ {printable}")
    startedAt = time.time()
    proc = subprocess.run(cmd, cwd=REPO)
    if proc.returncode != 0:
        raise SystemExit(f"\nFAILED (exit {proc.returncode}): {printable}\n"
                         f"Fix the cause, then re-run this script -- finished stages are skipped.")
    say(f"  done in {(time.time() - startedAt) / 60:.1f} min")


# state

def data_ready(workDir: Path) -> bool:
    preparedDir = workDir / "data/wmt14/prepared"
    return all((preparedDir / f"{split}.npz").exists() for split in ("train", "dev", "test")) and \
           (preparedDir / "vocab.src.json").exists()


def arm_done(workDir: Path, tag: str) -> bool:
    logPath = workDir / "runs/wmt14" / tag / "log.jsonl"
    if not logPath.exists():
        return False
    return any('"kind": "done"' in line for line in logPath.read_text().splitlines())


def arm_progress(workDir: Path, tag: str) -> str:
    logPath = workDir / "runs/wmt14" / tag / "log.jsonl"
    if not logPath.exists():
        return "not started"
    epochs = [json.loads(rawLine) for rawLine in logPath.read_text().splitlines()
              if rawLine.strip() and '"kind": "epoch"' in rawLine]
    if not epochs:
        return "started, no epoch finished yet"
    last = epochs[-1]
    state = "COMPLETE" if arm_done(workDir, tag) else "in progress"
    return f"{state}, {len(epochs)} epoch(s), dev ppl {last['dev_ppl']:.2f}"


def decode_done() -> bool:
    return (REPO / "results/wmt14_small/all_results.json").exists()


def status(workDir: Path) -> None:
    print(f"\nwork dir : {workDir}")
    print(f"repo     : {REPO}")
    print(f"{'data prepared':<22} {'yes' if data_ready(workDir) else 'NO'}")
    for tag, label in (("rev_seed1", "reversed arm"), ("fwd_seed1", "forward arm")):
        print(f"{label:<22} {arm_progress(workDir, tag)}")
    print(f"{'decoded + scored':<22} {'yes' if decode_done() else 'NO'}")
    resultsPath = REPO / "results/wmt14_small/results.md"
    if resultsPath.exists():
        print("\n--- results/wmt14_small/results.md ---")
        print(resultsPath.read_text())


# stages

SCALE_NOTE = ("0.5M training pairs vs the paper's 12M; 2x512 vs 4x1000; 32k/32k vocab vs "
              "160k/80k. Absolute BLEU is NOT comparable to Table 1 - the direction and "
              "shape of the effects are.")


def stage_data(workDir: Path, args) -> None:
    if data_ready(workDir) and not args.force:
        say("data already prepared - skipping")
        return
    rawDir = workDir / "data/wmt14/raw"
    run(["bash", "scripts/get_wmt14.sh"], args.dry_run)   # bash, not ./ - Drive strips +x
    if not (rawDir / "europarl.sub.en").exists() or args.force:
        run([sys.executable, "scripts/subsample_parallel.py",
             "--src-in", str(rawDir / "europarl.en"), "--tgt-in", str(rawDir / "europarl.fr"),
             "--src-out", str(rawDir / "europarl.sub.en"), "--tgt-out", str(rawDir / "europarl.sub.fr"),
             "--n", str(args.europarl_pairs), "--seed", "1"], args.dry_run)
    run([sys.executable, "scripts/prepare_data.py",
         "--out", str(workDir / "data/wmt14/prepared"),
         "--train-src", str(rawDir / "nc9.en"), str(rawDir / "europarl.sub.en"),
         "--train-tgt", str(rawDir / "nc9.fr"), str(rawDir / "europarl.sub.fr"),
         "--dev-src", str(rawDir / "newstest2013.en"), "--dev-tgt", str(rawDir / "newstest2013.fr"),
         "--test-src", str(rawDir / "newstest2014.en"), "--test-tgt", str(rawDir / "newstest2014.fr"),
         "--src-vocab-size", "32000", "--tgt-vocab-size", "32000",
         "--max-train-pairs", str(args.train_pairs), "--workers", str(args.workers),
         "--seed", "1"], args.dry_run)


def stage_bench(workDir: Path, args) -> None:
    run([sys.executable, "scripts/benchmark.py", "--corpus-pairs", str(args.train_pairs)],
        args.dry_run)


def _train(workDir: Path, args, reverse: bool, seed: int) -> None:
    tag = f"{'rev' if reverse else 'fwd'}_seed{seed}"
    if arm_done(workDir, tag) and not args.force:
        say(f"{tag} already complete ({arm_progress(workDir, tag)}) - skipping")
        return
    if (workDir / "runs/wmt14" / tag / "last.pt").exists():
        say(f"{tag}: resuming ({arm_progress(workDir, tag)})")
    run([sys.executable, "-m", "seq2seq.train", "--config", "configs/wmt14_small.yaml",
         "--reverse-source" if reverse else "--forward-source", "--seed", str(seed)],
        args.dry_run)


def stage_train_rev(workDir: Path, args) -> None:
    for split in args.seeds:
        _train(workDir, args, True, split)


def stage_train_fwd(workDir: Path, args) -> None:
    for split in args.seeds:
        _train(workDir, args, False, split)


def stage_decode(workDir: Path, args) -> None:
    if decode_done() and not args.force:
        say("results already present - skipping (use --force to redo)")
        return
    cmd = [sys.executable, "scripts/make_results.py", "--name", "wmt14_small",
           "--data-dir", str(workDir / "data/wmt14/prepared"),
           "--runs-root", str(workDir / "runs/wmt14"),
           "--out", "results/wmt14_small", "--beams", "1", "2", "12",
           "--seeds", *[str(split) for split in args.seeds],
           "--device", args.device, "--scale-note", SCALE_NOTE]
    if args.ensemble and len(args.seeds) > 1:
        cmd.append("--ensemble")
    run(cmd, args.dry_run)


def stage_report(workDir: Path, args) -> None:
    run([sys.executable, "scripts/make_report.py"], args.dry_run)


def stage_archive(workDir: Path, args) -> None:
    dest = workDir / "deliverables"
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
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=STAGES, help="run just these stages")
    parser.add_argument("--skip", nargs="+", choices=STAGES, default=["bench"],
                    help="stages to skip (default: bench)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--ensemble", action="store_true", help="also decode a seed ensemble")
    parser.add_argument("--train-pairs", type=int, default=500000)
    parser.add_argument("--europarl-pairs", type=int, default=1100000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="redo stages even if complete")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true", help="report progress and exit")
    args = parser.parse_args()

    workDir = work_dir()
    if args.status:
        status(workDir)
        return

    plan = args.only if args.only else [split for split in STAGES if split not in args.skip]
    say(f"work dir: {workDir}", rule=True)
    print(f"repo    : {REPO}")
    print(f"plan    : {' -> '.join(plan)}")
    print(f"seeds   : {args.seeds}\n")
    status(workDir)

    startedAt = time.time()
    for name in plan:
        say(f"STAGE: {name}", rule=True)
        RUNNERS[name](workDir, args)

    say(f"ALL DONE in {(time.time() - startedAt) / 3600:.2f} h", rule=True)
    status(workDir)


if __name__ == "__main__":
    main()
