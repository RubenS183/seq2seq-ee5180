# Reproducing Seq2Seq (Sutskever, Vinyals & Le, 2014)

Reversing the source sentence really does make translation better, and this repo shows it on a laptop. That is the paper's central claim and we reproduced it from scratch in PyTorch, then went one step further and removed the bottleneck the paper left behind.

This is an EE5180 course project in two halves. The mid-term reproduces rows 3 and 4 of the paper's Table 1, which is a single forward LSTM against a single reversed one. The end-term asks whether the paper's one fixed-length sentence vector is the thing actually holding the model back, and answers it by implementing Bahdanau attention on this same codebase and rerunning the identical grid.

**TA:** Prasenjit Kr Mudi (EE21D057)

## Read this before you read any number

The paper trained a 384M-parameter model, 4 layers of 1000 cells, on 12M sentence pairs across 8 GPUs for 10 days. We train 2 layers of 512 on 0.5M pairs on one Apple M1 Pro. So our absolute BLEU is nowhere near theirs and it is not supposed to be.

What we claim to reproduce is the **direction and shape** of the effects, not the scores. Think of it like checking that a spring obeys Hooke's law using a weaker spring. You get a different number for k but the same straight line. Every table below repeats this warning, because a BLEU of 6.6 next to the paper's 30.6 is meaningless unless you know why.

## What we got

### Mid-term: the reversal trick (WMT'14 En-Fr, scored on the paper's own ntst14, 3003 sentences)

| Model | beam | our BLEU (tokenized) | our BLEU (sacreBLEU) | our test ppl | paper BLEU |
|---|---|---|---|---|---|
| Single forward LSTM | 1 | 4.05 | 2.68 | 32.89 | - |
| Single forward LSTM | 2 | 4.24 | 2.86 | 32.89 | - |
| **Single forward LSTM** | **12** | **3.89** | 2.71 | 32.89 | 26.17 |
| Single reversed LSTM | 1 | 6.41 | 4.46 | 24.50 | - |
| Single reversed LSTM | 2 | 6.82 | 4.82 | 24.50 | - |
| **Single reversed LSTM** | **12** | **6.60** | 4.77 | 24.50 | 30.59 |

**What reproduced.** Reversal wins on both metrics, +2.71 BLEU and 25% lower test perplexity at beam 12. The paper got +4.42 BLEU and 19% lower perplexity, so we are the same shape and a bit smaller. The reversed model is also ahead at every single epoch of training and in every source-length bucket, so this is not a lucky checkpoint and it is not a short-sentence artefact.

**What did not reproduce, reported and not hidden.** The paper's BLEU climbs as the beam gets wider. Ours peaks at beam 2 and then falls at beam 12. We dug into why instead of just noting it, and the decomposition is in [`results/wmt14_small/beam_ablation.md`](results/wmt14_small/beam_ablation.md). A wider beam finds genuinely higher-probability translations that are worse translations, because at this scale the model's likelihood and actual quality have come apart. Length normalisation removes the brevity penalty and makes BLEU worse still, so brevity is a symptom and not the cause. This is the documented beam search curse and the paper's model, trained on 24x more data, was accurate enough to dodge it.

Absolute BLEU is also hurt because about 16% of our output tokens are `<unk>`, which is what a 32k vocabulary costs you.

### End-term: the bottleneck, and attention removing it

The mid-term model squeezes the whole source sentence into one fixed vector before decoding starts. At our scale that is 2,048 numbers and the paper's is 8,000. A 60-token sentence gets exactly the same room as a 6-token one, so the encoder has to decide what to keep before it knows what the decoder is going to ask for. That is the bottleneck.

WMT'14, reversed source, both arms trained identically:

| arm | beam | BLEU (tokenized) | test ppl | params |
|---|---|---|---|---|
| fixed vector (the paper's model) | 12 | **6.36** | 24.07 | 57.6M |
| + Bahdanau attention | 12 | **18.49** | 10.69 | 59.9M |

That is **+12.13 BLEU for 3.6% more parameters**, so you cannot wave it away as extra capacity. And the gain widens with sentence length exactly as the bottleneck story predicts, from +5.11 BLEU on 1-10 token sources to +15.20 on 51-70 token sources.

The perplexity probe says the same thing without a beam anywhere near it. The attention arm's perplexity is 0.65x the baseline's on short sentences and 0.32x on long ones, so the longer the source the more a single fixed vector costs you.

**Attention also makes the paper's trick redundant.** On Multi30k with 3 seeds per cell we ran the full 2x2 grid:

| direction | arm | BLEU (tokenized) | runs |
|---|---|---|---|
| forward | fixed vector | 4.72 ± 1.83 | 3 |
| forward | + attention | 23.51 ± 0.46 | 3 |
| reversed | fixed vector | 15.21 ± 0.63 | 3 |
| reversed | + attention | 25.73 ± 2.00 | 3 |

The reversal gap is +10.49 BLEU without attention and only +2.21 with it, so attention absorbs 79% of it. This is the clean explanation of what reversal was ever for. Reversing the source shortens the distance between a source word and the decoder step that needs it, and attention deletes that distance outright, so once you have attention there is much less left for reversal to buy.

**Where it does not show up.** On Multi30k the length curve does **not** widen with source length, and the attention arm actually falls off faster than the baseline there. We report that rather than quietly dropping the figure. Multi30k is 29k image captions with short sentences, so it has almost no long tail to test the claim on, and the WMT'14 curve is the better-powered test. The arm-level gain on Multi30k is real either way, it is just not explained by length.

### The bottleneck in miniature

The WMT evidence costs about 11 GPU-hours. There is a version that takes a minute and has no confounds at all, in [`results/copy_task/`](results/copy_task/).

Make the target equal to the source over uniformly random tokens. Nothing is memorisable and there is no language model to lean on, so emitting position *t* requires reading source position *t*. A decoder that only sees a fixed summary has to have stored every token in it, and attention just looks.

| arm | final loss per sequence | vs chance (28.6) |
|---|---|---|
| fixed vector | 4.875 | 17% |
| + Bahdanau attention | 0.004 | 0% |

Attention solves it completely and learns the exact diagonal alignment. The fixed vector does not solve it. That is the bottleneck on its own, with no corpus and no BLEU.

Full write-ups: [mid-term report](report/EE5180_midterm_report.pdf) and [end-term report](report/EE5180_endterm_report.pdf). Decks are in [`slides/`](slides/) and every table and figure is under [`results/`](results/).

## What we held exactly as the paper has it

This is the part that makes the reproduction mean anything, so it is worth being specific. Each row is auditable against the paper's text.

| Paper detail | Where |
|---|---|
| Two **separate** encoder and decoder LSTMs, no shared parameters | `src/seq2seq/model.py` |
| The encoder's final `(h, c)` over all layers **is** the sentence vector, and the decoder sees nothing else | `model.py` |
| Source reversed at train **and** test time, target never reversed | `data.py: collate()` |
| `<EOS>` terminates and is scored, and every OOV word becomes `UNK` | `data.py`, `vocab.py` |
| Uniform init `U(-0.08, 0.08)` on every parameter | `model.py: reset_parameters()` |
| Plain SGD at **lr 0.7**, no momentum, constant then **halved every half epoch** | `train.py: lr_at()` |
| Batch 128, loss is summed token NLL **divided by batch size**, then clip at 5 | `train.py` |
| Length-bucketed minibatches | `data.py: make_batches()` |
| Left-to-right beam search, a hypothesis **leaves the beam** on `<EOS>`, ranked by raw log-probability with no length normalisation | `beam.py` |
| Naive full softmax over the target vocabulary | `model.py` |

That loss convention is worth spelling out because it is easy to get wrong. The paper says to compute `s = ||g||` where g is the gradient divided by 128, and if `s > 5` set `g = 5g/s`. If you divide the loss by the number of **tokens** instead of by the **batch size** you quietly change what a clip threshold of 5 even means, so we divide by the batch size.

## Where we deviate, and why

| Paper | Ours | Why |
|---|---|---|
| 4 layers x 1000 cells, 1000-d embeddings, 384M params | 2 x 512, 512-d embeddings, ~58M params | 16 GB laptop |
| 160k source / 80k target vocabulary | 32k / 32k | the naive softmax dominates the cost |
| 12M selected WMT'14 pairs | 0.5M from News-Commentary v9 and Europarl v7 | compute budget |
| 7.5 epochs on 8 GPUs over ~10 days | ~8 epochs on 1 GPU in ~5 h per run | compute budget |
| Ensemble of 5 | up to 3 seeds, Multi30k only | wall-clock |
| `multi-bleu.pl` | sacreBLEU `--tokenize none`, which is equivalent, **and** standard sacreBLEU | reproducibility |

For the attention arm specifically we keep all three terms of Bahdanau's eq. 10 readout but use tanh in place of the maxout. Dropping the `E y_{i-1}` term would quietly turn this into Luong's readout rather than Bahdanau's, so we kept it. tanh bounds the readout, which means the output layer sees the same magnitudes as the baseline and Sutskever's lr 0.7 and clip 5 transfer across untouched. An unbounded maxout readout would have needed its own tuning, and then a BLEU gap between the arms could be an optimisation artefact instead of an architectural result. The encoder also stays unidirectional, because adding bidirectionality at the same time as attention would confound the two.

## Two BLEU numbers, always both

The paper reports cased BLEU from `multi-bleu.pl` on **tokenised** text. Modern practice is sacreBLEU on **detokenised** text. These are different numbers and they are not interchangeable, so every table reports both.

- `bleu_tok` is cased BLEU on Moses-tokenised hypothesis and reference, via `sacrebleu --tokenize none`. This is the `multi-bleu.pl` equivalent and it is the number comparable in kind to Table 1, so it is our headline.
- `bleu_detok` is sacreBLEU on detokenised output against raw references, with the signature recorded.

The test set is the paper's own. We evaluate on newstest2014, the full 3003-sentence version, pulled through `sacrebleu -t wmt14/full -l en-fr`, and dev is newstest2013. So our rows are scored on exactly the test set behind Table 1 and only the model and the training data are smaller.

## Getting the results back

### Setup

```bash
make setup
```

```bash
make test
```

`make test` runs 52 correctness gates in about 9 seconds. Do not skip it. These are the things that are cheap to check now and expensive to discover after an overnight run:

- beam search at `B=1` is **exactly** greedy decoding
- a wider beam never returns a hypothesis with lower log-probability
- reversing the input by hand and setting the reversal flag cancel out
- padding does not change the encoder's sentence vector
- scoring the references against themselves gives BLEU 100
- the memory-saving chunked loss matches the full one in value and in every gradient
- teacher-forced training and step-by-step decoding agree to 1e-6, so an off-by-one in the attention query cannot make training optimise a model that decoding does not implement
- **MPS and CPU agree on loss and gradients to 1e-3**, because PyTorch's Metal LSTM has historically produced silently wrong numbers and finding that out after an overnight run costs you a day

### The mid-term rows

```bash
make benchmark
```

```bash
make data-multi30k && make smoke && make results-multi30k
```

```bash
make data-wmt && make wmt && make results-wmt
```

```bash
make beam-ablation
```

`make data-wmt` is about a 700 MB download plus tokenisation. `make wmt` is the reported rows and takes roughly 8.5 h for both arms on an M1 Pro, then `make results-wmt` decodes at beam 1, 2 and 12, scores it and plots it in about 30 min.

Training is resumable. Every epoch writes `last.pt` atomically, so if a run dies you rerun the same command and it picks up from the last finished epoch.

### The end-term arms

```bash
make data-endterm
```

```bash
make endterm-multi30k && make endterm-wmt
```

```bash
make results-endterm-multi30k results-endterm-wmt && make alignments probe
```

Tier 0 on Multi30k is forward and reversed, times base and attention, times 3 seeds, in about 1.2 h. Tier 1 on WMT is the reversed arm both ways in about 10.5 h.

### Checks that prove the refactor behaved

```bash
make baseline-regression
```

```bash
make copy-task
```

`make baseline-regression` re-decodes the submitted mid-term checkpoints and diffs the hypotheses against the committed files. Decoding is deterministic so that diff has to come out **empty**, which is a far sharper statement than "BLEU still rounds to 6.60". It is what proves adding attention left the paper's model alone. `make copy-task` is the one-minute bottleneck demo above.

### If you do not want to train anything

A checkpoint is 220-229 MB and GitHub rejects anything over 100 MB, so the four checkpoints the gates need plus the evaluation splits are attached to the [`endterm-v1` release](https://github.com/RubenS183/seq2seq-ee5180/releases/tag/endterm-v1).

```bash
./scripts/fetch_artifacts.sh
```

That pulls 892 MB into `$EE5180_WORK` and checks every file against its SHA-256. Nothing we *report* needs it, because every number in the reports and the decks is committed under `results/`. It is what makes `make baseline-regression`, `make probe` and `make alignments` runnable from a fresh clone, so the numbers can be re-derived from the weights instead of just read.

### On a GPU

```bash
python scripts/colab_run_all.py
```

This does every WMT stage in order and you can rerun it after a disconnect, because finished stages are skipped and training resumes from the last completed epoch. There is a notebook wrapper in [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb).

## Three things that made this fit on a 16 GB laptop

The paper spent four of its eight GPUs on a naive softmax. At our scale that same softmax is still the thing that breaks first, so these are not micro-optimisations, they are the reason the run completes at all. All three are mathematically inert and the tests assert it.

**The logits tensor.** Batch 128 times 51 target positions times a 32k vocabulary is 836 MB for one tensor, and autograd keeps it alive for the backward pass. So `losses.py` computes the projection and the cross-entropy in checkpointed chunks over time, which means only one chunk's logits exist at once and the rest get recomputed during backward. On the worst-case batch, MPS memory fell from 5,676 MB to 3,528 MB and a step actually got faster, 0.94 s down to 0.75 s.

**Allocator growth.** Length bucketing produces many distinct tensor shapes and Metal's caching allocator keeps a block per shape, so memory creeps up until the machine starts swapping. Throughput collapsed from about 7,000 to under 1,000 target words a second partway through an epoch. Padded lengths are now rounded up to a multiple of 8 and the cache is released every 200 steps, which bounds the number of shapes. With both fixes the WMT run held about 7,150 target words a second, so 31 min an epoch and 4.2 h an arm.

**Decoding on the CPU, not the GPU.** Training runs on MPS but decoding defaults to CPU, which sounds backwards until you look at the shape of the work. Beam search is latency-bound rather than throughput-bound, because each step is one sentence times B beams, so Metal's per-kernel launch overhead dominates. Measured on the M1 Pro at `B=12`, it is **34 ms a sentence on CPU against 317 ms on MPS**, a 9x difference. Pass `--device mps` if you want to override it.

One more thing that cost real time to learn. Do not run a second heavy job alongside training. A concurrent multi-model decode pushed this machine into swap and slowed training down 5x.

Attention itself costs 1.18x the baseline, measured with the same instrument rather than assumed. The per-step attention loop is about 3x slower than the baseline's single fused LSTM call in isolation, but the decoder is only a small share of a step because the 32k softmax dominates, so it dilutes to 1.18x end to end.

## Layout

```
src/seq2seq/
  vocab.py       top-k word vocabulary and <unk> handling
  data.py        cleaning, Moses tokenisation, source reversal, length bucketing,
                 compact int32 corpus storage
  model.py       separate encoder and decoder LSTMs, U(-0.08, 0.08) init,
                 plus AttnDecoder and DecodeState behind `attention: bahdanau`
  attention.py   Bahdanau additive attention and the source mask
  losses.py      chunked checkpointed cross-entropy, which bounds the 32k softmax
  train.py       SGD 0.7 with the halving schedule, clip at 5, resumable checkpoints
  beam.py        left-to-right beam search and ensembling
  evaluate.py    decoding plus tokenized BLEU and sacreBLEU
  plots.py       BLEU-by-length, beam sweep, training curves, bottleneck overlay,
                 alignment heatmaps
  utils.py       device selection, seeding, checkpoint I/O
scripts/         data prep, the training grid, evaluation, the end-term probes,
                 report and slide generation, the Colab driver
configs/         multi30k.yaml, wmt14_small.yaml           (mid-term)
                 multi30k_endterm.yaml, wmt14_endterm.yaml (end-term)
tests/           52 correctness gates, run these before any long job
results/         reproduction tables, figures, decoded hypotheses, probes
report/          both write-ups, .md and .pdf
slides/          both decks
notebooks/       colab_run.ipynb, the same code on a Colab GPU
```

Heavy artifacts, meaning the venv, the corpora and the checkpoints, live in `$EE5180_WORK` which defaults to `~/ee5180-work`. That is deliberately **outside** the iCloud-synced course folder, because a training run that checkpoints into iCloud thrashes sync and can get its files evicted mid-run.

The whole codebase is written from the two papers' text with no reference repository, which is 1,766 lines across 11 modules. That is on purpose, because the encoder and decoder split, the reversal, the SGD schedule, the clipping convention and the beam search are all things a marker can read against the paper and check.

## Work split

| Member | Owns | Files |
|---|---|---|
| M1, data | download, cleaning, tokenisation, vocab and `UNK`, reversal, bucketing | `vocab.py`, `data.py`, `scripts/get_*.sh`, `subsample_parallel.py`, `prepare_data.py` |
| M2, model and training | encoder and decoder, loss, SGD schedule, clipping, checkpointing | `model.py`, `losses.py`, `train.py`, `utils.py`, `benchmark.py`, `colab_run_all.py` |
| M3, decoding and eval | beam search, ensembling, the BLEU harness | `beam.py`, `evaluate.py`, `analyze_beam.py` |
| M4, analysis and writing | the grid, plots, reports, slides | `plots.py`, `make_results.py`, `make_endterm_results.py`, `plot_alignments.py`, `probe_bottleneck.py`, `copy_task.py`, `make_figures.py`, `make_report.py`, `md_to_pdf.py`, `make_slides.js` |

## Still out of scope

The Transformer contrast stays prose and we did not do SMT 1000-best rescoring, which is the paper's Table 2. Both need compute we do not have.

## References

- Sutskever, Vinyals & Le. *Sequence to Sequence Learning with Neural Networks.* NeurIPS 2014. [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)
- Bahdanau, Cho & Bengio. *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR 2015. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
