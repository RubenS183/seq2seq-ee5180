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
- **`make test`** runs the correctness gates in about twelve seconds, including beam(B=1)≡greedy and MPS≡CPU gradient agreement.

## Work split

| Member | Owns | Files |
|---|---|---|
| M1 — Data | download, cleaning, tokenisation, vocab/`UNK`, reversal, bucketing | `vocab.py`, `data.py`, `scripts/get_*.sh`, `prepare_data.py`, `subsample_parallel.py` |
| M2 — Model & training | encoder/decoder, loss, SGD schedule, clipping, checkpointing | `model.py`, `train.py`, `utils.py`, `benchmark.py` |
| M3 — Decoding & eval | beam search, ensembling, BLEU harness | `beam.py`, `evaluate.py` |
| M4 — Analysis & writing | grid, plots, report, slides | `plots.py`, `make_results.py`, `make_figures.py`, `make_report.py`, `make_slides.js` |
