# EE5180 Seq2Seq reproduction. Heavy artifacts live in $(WORK), outside iCloud.
WORK ?= $(HOME)/ee5180-work
PY   ?= $(WORK)/.venv/bin/python
export EE5180_WORK = $(WORK)

.PHONY: help setup test data-multi30k data-wmt smoke wmt results-multi30k results-wmt clean-runs

help:
	@grep -E '^[a-z0-9-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

setup:              ## create the venv outside iCloud and install pinned deps
	uv venv --python 3.12 $(WORK)/.venv
	VIRTUAL_ENV=$(WORK)/.venv uv pip install -r requirements.txt

test:               ## correctness gates (incl. MPS-vs-CPU agreement)
	$(PY) -m pytest tests/test_smoke.py -q

data-multi30k:      ## download + prepare Multi30k En-Fr (Tier 0)
	./scripts/get_multi30k.sh
	$(PY) scripts/prepare_data.py --out $(WORK)/data/multi30k/prepared \
	  --train-src $(WORK)/data/multi30k/raw/train.en --train-tgt $(WORK)/data/multi30k/raw/train.fr \
	  --dev-src   $(WORK)/data/multi30k/raw/val.en   --dev-tgt   $(WORK)/data/multi30k/raw/val.fr \
	  --test-src  $(WORK)/data/multi30k/raw/test_2016_flickr.en \
	  --test-tgt  $(WORK)/data/multi30k/raw/test_2016_flickr.fr \
	  --src-vocab-size 10000 --tgt-vocab-size 10000

data-wmt:           ## download + prepare the WMT'14 En-Fr subset (Tier 1)
	./scripts/get_wmt14.sh
	$(PY) scripts/subsample_parallel.py \
	  --src-in $(WORK)/data/wmt14/raw/europarl.en --tgt-in $(WORK)/data/wmt14/raw/europarl.fr \
	  --src-out $(WORK)/data/wmt14/raw/europarl.sub.en --tgt-out $(WORK)/data/wmt14/raw/europarl.sub.fr \
	  --n 1100000 --seed 1
	$(PY) scripts/prepare_data.py --out $(WORK)/data/wmt14/prepared \
	  --train-src $(WORK)/data/wmt14/raw/nc9.en $(WORK)/data/wmt14/raw/europarl.sub.en \
	  --train-tgt $(WORK)/data/wmt14/raw/nc9.fr $(WORK)/data/wmt14/raw/europarl.sub.fr \
	  --dev-src   $(WORK)/data/wmt14/raw/newstest2013.en --dev-tgt $(WORK)/data/wmt14/raw/newstest2013.fr \
	  --test-src  $(WORK)/data/wmt14/raw/newstest2014.en --test-tgt $(WORK)/data/wmt14/raw/newstest2014.fr \
	  --src-vocab-size 32000 --tgt-vocab-size 32000 --max-train-pairs 500000

benchmark:          ## measure tgt-words/s to size the WMT run
	$(PY) scripts/benchmark.py

smoke:              ## Tier 0 grid: {fwd,rev} x 3 seeds on Multi30k
	./scripts/run_grid.sh configs/multi30k.yaml 1 2 3

wmt:                ## Tier 1: the reported Table 1 rows (long; run overnight)
	./scripts/run_grid.sh configs/wmt14_small.yaml 1

results-multi30k:   ## decode + score + plot Tier 0
	$(PY) scripts/make_results.py --name multi30k \
	  --data-dir $(WORK)/data/multi30k/prepared --runs-root $(WORK)/runs/multi30k \
	  --out results/multi30k --beams 1 2 12 --seeds 1 2 3 --ensemble \
	  --scale-note "Multi30k (29k image captions), 2x256 LSTM. Pipeline validation and a seed-controlled ablation - NOT the paper's corpus or test set."

results-wmt:        ## decode + score + plot Tier 1 (the submitted table)
	$(PY) scripts/make_results.py --name wmt14_small \
	  --data-dir $(WORK)/data/wmt14/prepared --runs-root $(WORK)/runs/wmt14 \
	  --out results/wmt14_small --beams 1 2 12 --seeds 1 \
	  --scale-note "0.5M pairs vs the paper's 12M; 2x512 vs 4x1000; 32k/32k vocab vs 160k/80k; ~8 epochs on one M1 Pro GPU vs 7.5 epochs on 8 GPUs for 10 days. Absolute BLEU is NOT comparable to Table 1 - the direction and shape of the effects are."

clean-runs:
	rm -rf $(WORK)/runs
