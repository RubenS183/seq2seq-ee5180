## Why a wider beam hurts at our scale

The paper's BLEU rises with beam size. Ours peaks at beam 2. Decomposing BLEU shows
why, and shows that length normalisation is *not* the fix.

| ranking | beam | BLEU | brevity penalty | length ratio | 1-gram prec | 4-gram prec |
|---|---|---|---|---|---|---|
| raw log-probability (paper's setting) | 1 | **6.41** | 0.979 | 0.979 | 36.07 | 1.44 |
| raw log-probability (paper's setting) | 2 | **6.82** | 0.966 | 0.967 | 36.39 | 1.69 |
| raw log-probability (paper's setting) | 12 | **6.60** | 0.943 | 0.945 | 34.75 | 1.75 |
| length-normalised (alpha=1) | 1 | **6.41** | 0.979 | 0.979 | 36.07 | 1.44 |
| length-normalised (alpha=1) | 2 | **6.82** | 1.000 | 1.024 | 35.07 | 1.63 |
| length-normalised (alpha=1) | 12 | **6.07** | 1.000 | 1.143 | 30.74 | 1.49 |

**Reading it.**

- With the paper's raw-log-probability ranking, widening the beam shortens the output: length ratio 0.979 -> 0.945 and the brevity penalty 0.979 -> 0.943. Every extra token lowers a sequence's total log-probability, so a larger beam prefers shorter hypotheses.
- Length normalisation removes the brevity penalty entirely (ratio 1.143, BP 1.000) but **does not fix BLEU**: it overshoots into over-long output and 1-gram precision falls 34.75 -> 30.74, so BLEU drops further (6.60 -> 6.07).
- So the cause is not calibration of length alone. A wider beam genuinely finds *higher-probability* hypotheses -- 4-gram precision rises with beam under the raw ranking -- but at this scale the model's likelihood and translation quality have diverged, so searching harder optimises the wrong thing. This is the documented 'beam search curse'; the paper's model, trained on 24x more data, was accurate enough not to suffer it.
- Roughly **16% of our output tokens are `<unk>`** (12,176 of 76,716), a direct consequence of the 32k vocabulary. The paper notes its own BLEU was penalised on out-of-vocabulary words at 80k; ours is penalised far harder, and that is a large part of the absolute gap.
