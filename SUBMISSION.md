# What to submit — EE5180 mid-term

Everything below is produced by the commands in `README.md`; nothing is hand-edited.

## Hand in

| File | What it is |
|---|---|
| `report/EE5180_midterm_report.pdf` | The mid-term report: method walkthrough, declared deviations, the reproduction table, figures, and what we do and do not claim. |
| `slides/EE5180_midterm_slides.pptx` | The mid-term presentation (12 slides, speaker notes included). |
| `results/wmt14_small/` | The reproduced Table 1 rows: `results.md`, `table1_reproduction.csv`, `all_results.json`, figures, decoded hypotheses. |
| `results/multi30k/` | The seed-controlled supporting ablation. |
| The repository itself | Code, configs, tests, and the exact commands to regenerate all of the above. |

If a single archive is wanted: `make submit` writes `EE5180_seq2seq_submission.zip`
containing the repository without the venv, corpora or checkpoints.

## Where each requirement is answered

| Requirement (topic slides, pp. 8–9) | Where |
|---|---|
| Reproduce 1–2 rows of Table 1 | `results/wmt14_small/results.md`, report §5. Rows 3 and 4 — single forward vs single reversed LSTM — decoded at beam 1, 2 and 12 and scored on the paper's own ntst14. |
| Demonstrate understanding of the method | Report §2–§3, slides 3–6. Architecture and time-lag figures, every paper hyperparameter we hold fixed, and every deviation declared with its reason. |
| PyTorch reference repo permitted | Not used — implemented from scratch in `src/seq2seq/`, so the encoder/decoder split, reversal, SGD schedule, clipping convention and beam search are all auditable against the paper's text. |

## Points a marker may want to check quickly

- **The reversal flag is the only difference** between the two reported rows: same data, same seed, same batch order, same schedule (`scripts/run_grid.sh`).
- **The test set is the paper's own** — newstest2014, full 3003-sentence version, fetched via sacreBLEU (`scripts/get_wmt14.sh`).
- **Both BLEU conventions are reported**, with signatures: tokenized cased BLEU (the `multi-bleu.pl` equivalent the paper used) and standard sacreBLEU on detokenized output.
- **The scale gap is stated next to every number**, and we never present our BLEU as comparable to the paper's.
- **`make test`** runs 23 correctness gates in a few seconds, including beam(B=1)≡greedy, MPS≡CPU gradient agreement, and exact loss/gradient equality for the memory-saving chunked loss.

## Work split

| Member | Owns | Files |
|---|---|---|
| M1 — Data | download, cleaning, tokenisation, vocab/`UNK`, reversal, bucketing | `vocab.py`, `data.py`, `scripts/get_*.sh`, `prepare_data.py`, `subsample_parallel.py` |
| M2 — Model & training | encoder/decoder, loss, SGD schedule, clipping, checkpointing | `model.py`, `losses.py`, `train.py`, `utils.py`, `benchmark.py`, `colab_run_all.py` |
| M3 — Decoding & eval | beam search, ensembling, BLEU harness | `beam.py`, `evaluate.py`, `analyze_beam.py` |
| M4 — Analysis & writing | grid, plots, report, slides | `plots.py`, `make_results.py`, `make_figures.py`, `make_report.py`, `make_slides.js` |


---

# What to submit — EE5180 end-term

Everything below is produced by the `endterm-*` targets in the `Makefile`;
nothing is hand-edited.

## Hand in

| File | What it is |
|---|---|
| `report/EE5180_endterm_report.pdf` | The end-term report: the bottleneck, the corpus change, Bahdanau attention and its declared deviations, the controlled comparison, the direct probes, and the Attention → Transformer line. |
| `slides/EE5180_endterm_slides.pptx` | The end-term presentation (17 slides, speaker notes included). |
| `results/wmt14_endterm/` | The headline comparison: `endterm_results.md`, both arms' tables, the BLEU-by-length overlay, alignment heatmaps, the perplexity and saturation probes. |
| `results/multi30k_endterm/` | The seed-controlled 2×2 grid (`{forward,reversed} × {fixed vector, attention}`, 3 seeds). |
| `results/copy_task/` | The bottleneck in miniature — a task the fixed vector cannot solve and attention can. |
| The repository itself | Code, configs, 52 tests, and the exact commands to regenerate all of the above. |

`make submit-endterm` writes `EE5180_seq2seq_endterm.zip` (repository without the
venv, corpora or checkpoints).

## Where each requirement is answered

| Requirement (topic slides, pp. 8–9) | Where |
|---|---|
| Study the bottleneck problem of the fixed-length vector | Report §1 and §7, slides 4 and 15. Quantified three ways that do not share a failure mode: BLEU by source-length bucket, per-token perplexity by bucket (no beam search involved), and sentence-vector norm saturation. |
| Explain how it motivated Attention | Report §3 and §5–6, slides 6 and 11–13. Not only explained — **implemented** (`src/seq2seq/attention.py`, `AttnDecoder` in `model.py`) and measured against a baseline retrained on identical data, with the identical grid rerun. |
| … and how Attention motivated the Transformer | Report §9, slide 16. The argument is grounded in this project's own measurements: the constant-length path, the per-step loop we had to pay for because the context enters the recurrence, and the `(B,T,S,A)` memory wall that makes the dot-product score necessary rather than cosmetic. |
| Time permitting: roadmap to modern LLM-based MT | Report §9 closing paragraph. Traced, but we do not claim to have tested the Transformer step — the self-attention run stayed out of scope and is named as such. |

## Points a marker may want to check quickly

- **The length-mismatch filter is gone.** `clean_pairs`'s ratio test was raised in
  review of the mid-term. The end-term corpus is prepared with `--max-ratio inf`
  and the drop report at
  `$EE5180_WORK/data/wmt14/prepared_nofilter/prepare_report.json` records
  `"ratio": 0`. The 50-token cap is lifted to 100 as well, which matters more:
  it had been removing 13.3% of the corpus, and long sentences are exactly where
  a fixed-length vector is supposed to fail.
- **The training-set size is held fixed** at 500,000 pairs, so the corpus *size*
  is constant and only its length distribution changes (mean target length
  26.07 → 30.80 tokens).
- **The baseline was retrained** on that corpus
  (`runs/wmt14_endterm/rev_seed1`). Comparing attention-on-new-data against the
  mid-term's no-attention-on-old-data would confound architecture with corpus.
- **`attention: bahdanau` is the only difference between the arms** — one config
  file, same seed, batch order, optimiser, schedule, clipping and init.
  Parameters differ by 3.6%, so the gap is not extra capacity.
- **The mid-term is provably untouched.** `make baseline-regression` re-decodes
  the submitted checkpoints and `diff`s against the committed hypotheses;
  decoding is deterministic so the diff is empty, on all 3,003 sentences, both
  directions. The mid-term's 23 tests still pass unmodified.
- **`make test`** runs 52 gates in ~14 s, including attention weights being
  exactly zero on padding, teacher-forced training agreeing with step-by-step
  decoding to 1e-6, beam(B=1) ≡ greedy for the attention arm, and MPS ≡ CPU
  gradients over the new Metal kernels.
- **Cost is measured, not asserted** — `make benchmark` reports both arms with
  the same instrument: 1.18× slower, +3.6% parameters.
- **The scale gap is stated next to every number**, and we never present our BLEU
  as comparable to the paper's.

## Work split — end-term additions

| Member | Owns | Files |
|---|---|---|
| M1 — Data | the unfiltered corpus and its drop report | `scripts/prepare_data.py`, the `data-endterm` target |
| M2 — Model & training | attention module, attention decoder, decode-state plumbing, throughput | `attention.py`, `model.py`, `benchmark.py`, `configs/*_endterm.yaml` |
| M3 — Decoding & eval | beam search over both architectures, length buckets, the byte-exact baseline gate | `beam.py`, `evaluate.py`, `scripts/baseline_regression.sh` |
| M4 — Analysis & writing | the two-arm join, alignment heatmaps, probes, copy task, report, slides | `make_endterm_results.py`, `plot_alignments.py`, `probe_bottleneck.py`, `copy_task.py`, `plots.py`, `make_endterm_report.py`, `make_endterm_slides.js` |
