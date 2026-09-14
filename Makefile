# EE5180 Seq2Seq reproduction. Heavy artifacts live in $(WORK), outside iCloud.
WORK ?= $(HOME)/ee5180-work
PY   ?= $(WORK)/.venv/bin/python
export EE5180_WORK = $(WORK)

.PHONY: help setup test data-multi30k data-wmt smoke wmt results-multi30k results-wmt beam-ablation figures report slides submit clean-runs

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

beam-ablation:      ## why a wider beam hurts: length-normalised decode + BLEU decomposition
	for B in 1 2 12; do \
	  PYTHONPATH=src $(PY) -m seq2seq.evaluate \
	    --checkpoint $(WORK)/runs/wmt14/rev_seed1/best.pt \
	    --data-dir $(WORK)/data/wmt14/prepared --beam $$B --reverse-source \
	    --length-norm 1.0 --out-dir results/wmt14_small/lengthnorm --tag rev_seed1_ln > /dev/null; \
	done
	$(PY) scripts/analyze_beam.py

figures:            ## regenerate the explanatory figures
	$(PY) scripts/make_figures.py results/figures

report:             ## rebuild the mid-term report (.md and .pdf) from the results
	$(PY) scripts/make_report.py
	$(PY) scripts/md_to_pdf.py report/EE5180_midterm_report.md

slides:             ## rebuild the mid-term slide deck from the results
	node scripts/make_slides.js

submit:             ## package the repository for submission
	$(PY) scripts/make_report.py && $(PY) scripts/md_to_pdf.py report/EE5180_midterm_report.md
	node scripts/make_slides.js
	rm -f EE5180_seq2seq_submission.zip
	zip -qr EE5180_seq2seq_submission.zip . \
	  -x '*.git/*' '*__pycache__/*' '*.pytest_cache/*' '*.DS_Store' '*.pt' 'EE5180_seq2seq_submission.zip'
	@echo "wrote EE5180_seq2seq_submission.zip ($$(du -h EE5180_seq2seq_submission.zip | cut -f1))"

clean-runs:
	rm -rf $(WORK)/runs
