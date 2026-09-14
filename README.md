# Reproducing Sutskever, Vinyals & Le (2014) — *Sequence to Sequence Learning with Neural Networks*

EE5180 course project, mid-term deliverable. **TA:** Prasenjit Kr Mudi (EE21D057).

This repository reproduces **rows 3 and 4 of Table 1** of the paper — a single
*forward* LSTM against a single *reversed* LSTM on WMT'14 English→French — at a
scale that fits one laptop GPU, plus the beam sweep, the seed ablation and the
BLEU-versus-sentence-length analysis.

> **Read this before reading any number below.** The paper trained a 384M-parameter,
> 4×1000 LSTM on 12M sentence pairs across 8 GPUs for 10 days. We train a
> 2×512 LSTM on a 0.5M-pair subset on a single Apple M1 Pro. **Our absolute BLEU
> is not comparable to the paper's.** What we claim to reproduce is the
> *direction and shape* of the paper's effects. Every results table in this repo
> repeats this warning next to the numbers.

---

## Results

> **Scale gap, stated first.** 0.5M training pairs against the paper's 12M; 2×512
> against 4×1000; a 32k/32k vocabulary against 160k/80k; one laptop GPU against eight.
> Compare forward against reversed **within our column** — never against the paper's.

### Table 1 rows 3–4 — WMT'14 En→Fr, scored on the paper's own ntst14 (3,003 sentences)

| Model | beam | BLEU (tokenized) | BLEU (sacreBLEU) | test ppl | paper BLEU |
|---|---|---|---|---|---|
| Single forward LSTM | 1 | 4.05 | 2.68 | 32.89 | — |
| Single forward LSTM | 2 | 4.24 | 2.86 | 32.89 | — |
| **Single forward LSTM** | **12** | **3.89** | 2.71 | 32.89 | 26.17 |
| Single reversed LSTM | 1 | 6.41 | 4.46 | 24.50 | — |
| Single reversed LSTM | 2 | 6.82 | 4.82 | 24.50 | — |
| **Single reversed LSTM** | **12** | **6.60** | 4.77 | 24.50 | 30.59 |

**Reproduced**

- **Reversal wins on both metrics:** +2.71 BLEU and 25% lower test
  perplexity at beam 12 (paper: +4.42 BLEU, 19% lower perplexity).
- The reversed model is ahead at **every epoch** of training and in **every source-length
  bucket**, so the gap is neither a lucky checkpoint nor a short-sentence artefact.
- On Multi30k, three seeds per arm make the effect far larger than seed noise, and a
  3-seed reversed ensemble adds +2.7 BLEU over a single model
  (16.01 → 18.69 at beam 2).

**Did not reproduce — reported, not hidden**

- **Monotone gains from a wider beam.** BLEU peaks at beam 2 and falls at beam 12.
  Length normalisation removes the brevity penalty yet lowers BLEU further, so brevity is a
  symptom: a wider beam finds likelier but worse translations
  ([`results/wmt14_small/beam_ablation.md`](results/wmt14_small/beam_ablation.md)).
- **Robustness on long sentences.** Both models lose most of their BLEU beyond 40 source
  tokens — the fixed-vector bottleneck, visible at our scale, and the end-term's subject.
- **Absolute BLEU.** Beyond the data gap, ~16% of output tokens are `<unk>` under a 32k
  vocabulary.

Full write-up: [`report/EE5180_midterm_report.pdf`](report/EE5180_midterm_report.pdf) ·
slides: [`slides/EE5180_midterm_slides.pptx`](slides/EE5180_midterm_slides.pptx) ·
tables: [`results/`](results/).

## The claim under test

The paper's central empirical contribution (sec. 3.3) is that **reversing the
word order of the source sentence — and not the target — markedly improves
translation**, moving test perplexity 5.8 → 4.7 and BLEU 25.9 → 30.6, because
reversal shortens the *minimal time lag* between the first source words and the
first target words and so makes the optimisation problem easier.

In our grid the reversal flag is the **only** difference between the two
reported rows: identical data, identical vocabularies, identical seed, identical
batch order, identical schedule.

## What is held exactly as in the paper

| Paper detail | Where |
|---|---|
| Two **separate** encoder/decoder LSTMs, no shared parameters | `src/seq2seq/model.py` |
| Encoder's final `(h, c)` over all layers *is* the sentence vector; decoder sees nothing else | `model.py` |
| Source reversed at train **and** test; target never reversed | `data.py: collate()` |
| `<EOS>` terminates and is scored; OOV → `UNK` | `data.py`, `vocab.py` |
| Uniform init `U(−0.08, 0.08)` | `model.py: reset_parameters()` |
| Plain SGD, **lr 0.7**, no momentum; constant then **halved every half epoch** | `train.py: lr_at()` |
| Batch 128, loss = summed token NLL **÷ batch size**, then `clip_grad_norm_(·, 5)` | `train.py` |
| Length-bucketed minibatches | `data.py: make_batches()` |
| Left-to-right beam search; a hypothesis **leaves the beam** on `<EOS>`; ranked by raw log-probability, no length normalisation | `beam.py` |
| Naive full softmax over the target vocabulary | `model.py` |

The loss normalisation is worth spelling out: the paper says *"compute
s = ‖g‖₂ where g is the gradient divided by 128; if s > 5, set g = 5g/s."*
Dividing the loss by the number of **tokens** instead of by the **batch size**
would silently change what the clip threshold of 5 means, so we divide by the
batch size.

## Declared deviations

| Paper | Ours | Why |
|---|---|---|
| 4 layers × 1000 cells, 1000-d emb, 384M params | 2 × 512, 512-d emb, ~58M params | 16 GB laptop |
| 160k source / 80k target vocabulary | 32k / 32k | the naive softmax dominates cost |
| 12M "selected" WMT'14 pairs | 0.5M from News-Commentary v9 + Europarl v7 | compute budget |
| 7.5 epochs, 8 GPUs, ~10 days | ~8 epochs, 1 GPU, ~5 h per run | compute budget |
| Ensemble of 5 | up to 3 seeds (Tier 0 only) | wall-clock |
| `multi-bleu.pl` | sacreBLEU `--tokenize none` (equivalent) **and** standard sacreBLEU | reproducibility |

## Evaluation — two numbers, always both

The paper reports **cased BLEU from `multi-bleu.pl` on tokenised text**.
Modern practice is sacreBLEU on **detokenised** text. These are different
numbers and are not interchangeable, so every table reports both:

* `bleu_tok` — cased BLEU on Moses-tokenised hypothesis/reference
  (`sacrebleu --tokenize none`). This is the `multi-bleu.pl` equivalent and the
  number comparable *in kind* to Table 1. **Headline.**
* `bleu_detok` — sacreBLEU on detokenised output vs. raw references, signature recorded.

**Test set.** Tier 1 evaluates on **newstest2014, full 3003-sentence version** —
the paper's own `ntst14` — pulled through `sacrebleu -t wmt14/full -l en-fr`.
Dev is newstest2013. So our rows are scored on exactly the test set behind
Table 1; only the model and training data are smaller.

## Layout

```
src/seq2seq/
  vocab.py       top-k word vocabulary, <unk> handling
  data.py        cleaning, Moses tokenisation, source reversal, length bucketing,
                 compact int32 corpus storage
  model.py       separate encoder / decoder LSTMs, U(-0.08, 0.08) init
  losses.py      chunked, checkpointed cross-entropy (bounds the 32k-softmax memory)
  train.py       SGD 0.7 with halving schedule, clip at 5, resumable checkpoints
  beam.py        left-to-right beam search, ensembling
  evaluate.py    decoding plus tokenized BLEU and sacreBLEU
  plots.py       BLEU-by-length (Fig. 3), beam sweep, training curves
  utils.py       device selection, seeding, checkpoint I/O
scripts/
  get_multi30k.sh, get_wmt14.sh, subsample_parallel.py, prepare_data.py   data
  benchmark.py, run_grid.sh                                               training
  make_results.py, analyze_beam.py                                        evaluation
  make_figures.py, make_report.py, md_to_pdf.py, make_slides.js           write-up
  colab_run_all.py                                                        one-shot Colab driver
configs/         multi30k.yaml (Tier 0), wmt14_small.yaml (Tier 1)
tests/           23 correctness gates — run these before any long job
results/         reproduction tables, figures, decoded hypotheses, beam ablation
report/          mid-term write-up (.md and .pdf)
slides/          mid-term presentation (.pptx)
notebooks/       colab_run.ipynb — the same code on a Colab GPU
```

Heavy artifacts (venv, corpora, checkpoints) live in `$EE5180_WORK`
(default `~/ee5180-work`), deliberately **outside** the iCloud-synced course
folder: a training run checkpointing into iCloud thrashes sync and risks
eviction mid-run.

## Work split

| Member | Owns | Files |
|---|---|---|
| M1 — Data | download, cleaning, tokenisation, vocab/`UNK`, reversal, bucketing | `vocab.py`, `data.py`, `scripts/get_*.sh`, `subsample_parallel.py`, `prepare_data.py` |
| M2 — Model & training | encoder/decoder, loss, SGD schedule, clipping, checkpointing | `model.py`, `losses.py`, `train.py`, `utils.py`, `benchmark.py`, `colab_run_all.py` |
| M3 — Decoding & eval | beam search, ensembling, BLEU harness | `beam.py`, `evaluate.py`, `analyze_beam.py` |
| M4 — Analysis & writing | grid, plots, report, slides | `plots.py`, `make_results.py`, `make_figures.py`, `make_report.py`, `md_to_pdf.py`, `make_slides.js` |

## Reproducing from scratch

```bash
make setup            # venv outside iCloud + pinned deps
make test             # 23 correctness gates, a few seconds — do not skip
make benchmark        # measures tgt-words/s; sizes the WMT run

make data-multi30k    # Tier 0 corpus (seconds)
make smoke            # {fwd,rev} x 3 seeds, ~35 min on an M1 Pro
make results-multi30k

make data-wmt         # ~700 MB download + tokenisation
make wmt              # the reported rows — ~8.5 h for both arms on an M1 Pro
make results-wmt      # decode beam 1/2/12, score, plot (~30 min)
make beam-ablation    # length-normalised decode + BLEU decomposition (~30 min)

make figures report slides   # rebuild the explanatory figures, PDF report and deck
```

On a GPU, `python scripts/colab_run_all.py` does every WMT stage in order and can be
re-run after a disconnect: finished stages are skipped and training resumes from the
last completed epoch. See `notebooks/colab_run.ipynb`.

Training is resumable: every epoch writes `last.pt` atomically, and a rerun of
the same command picks up where it stopped.

**Memory: two fixes that made the run fit on a 16 GB Mac.** The paper spent four of its
eight GPUs on a naive softmax; at our scale the same softmax was still the bottleneck.

- *The logits tensor.* Batch 128 × 51 target positions × 32k vocabulary is 836 MB for
  one tensor, and autograd keeps it alive for the backward pass. `losses.py` computes
  the projection and cross-entropy in checkpointed chunks over time
  (`loss_chunk_size: 8`). On the worst-case batch, MPS memory fell from 5,676 MB to
  3,528 MB and a step got faster (0.94 → 0.75 s).
- *Allocator growth.* Length bucketing produces many distinct tensor shapes, and Metal's
  caching allocator keeps a block per shape; throughput collapsed from ~7,000 to under
  1,000 target words/s partway through an epoch. Padded lengths are now rounded up to a
  multiple of 8 and the cache is released every 200 steps (`empty_cache_every`).

Both are mathematically inert, and tests assert it: the chunked loss matches the full
one in value and in every gradient, and the extra padding leaves the loss unchanged.
With them the WMT run held ~7,150 target words/s: 31 min per epoch, 4.2 h per arm.

Do not run a second heavy job alongside training. A concurrent multi-model decode
pushed this machine into swap and slowed training 5×.

**Device note.** Training runs on MPS; **decoding defaults to CPU**. Beam search
is latency-bound rather than throughput-bound — each step is one sentence × B
beams — so Metal's per-kernel launch overhead dominates. Measured on the M1 Pro
at B=12: **34 ms/sentence on CPU vs 317 ms/sentence on MPS**, a 9× difference.
Pass `--device mps` to override.

### Correctness gates

`make test` asserts, among others:

* beam search with `B=1` is **exactly** greedy decoding;
* a wider beam never returns a hypothesis with lower log-probability;
* reversing the input by hand and setting the reversal flag cancel out;
* padding does not change the encoder's sentence vector;
* scoring the references against themselves gives BLEU 100;
* **MPS and CPU agree on loss and gradients to 1e-3** — PyTorch's Metal LSTM
  has historically produced silently wrong numbers, and finding that out after
  an overnight run costs a day.

## Scope

Mid-term only. Attention, the Transformer contrast, and SMT 1000-best rescoring
(Table 2) are end-term work; `model.py` is structured so Bahdanau attention
drops in behind a flag and the identical grid reruns.
