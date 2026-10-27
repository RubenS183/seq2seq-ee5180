# EE5180 Seq2Seq reproduction. Heavy artifacts live in $(WORK), outside iCloud.
WORK ?= $(HOME)/ee5180-work
PY   ?= $(WORK)/.venv/bin/python
export EE5180_WORK = $(WORK)

.PHONY: help setup test data-multi30k data-wmt smoke wmt results-multi30k results-wmt beam-ablation figures report slides submit clean-runs \
        baseline-regression copy-task data-endterm endterm-multi30k endterm-wmt \
        results-endterm-multi30k results-endterm-wmt alignments probe endterm-figures \
        endterm-report endterm-slides submit-endterm

help:
	@grep -E '^[a-z0-9-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

setup:              ## create the venv outside iCloud and install pinned deps
	uv venv --python 3.12 $(WORK)/.venv
	VIRTUAL_ENV=$(WORK)/.venv uv pip install -r requirements.txt

test:               ## correctness gates (incl. MPS-vs-CPU agreement, both arms)
	$(PY) -m pytest tests -q

baseline-regression: ## did attention change the paper's model? byte-exact answer
	./scripts/baseline_regression.sh

copy-task:          ## the bottleneck in miniature (~1 min, no corpus needed)
	$(PY) scripts/copy_task.py --out results/copy_task

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

data-endterm:       ## prepare BOTH corpora with no length-mismatch filtering
	@echo "no ratio filter (--max-ratio inf), length cap raised 50 -> 100"
	$(PY) scripts/prepare_data.py --out $(WORK)/data/multi30k/prepared_nofilter \
	  --train-src $(WORK)/data/multi30k/raw/train.en --train-tgt $(WORK)/data/multi30k/raw/train.fr \
	  --dev-src   $(WORK)/data/multi30k/raw/val.en   --dev-tgt   $(WORK)/data/multi30k/raw/val.fr \
	  --test-src  $(WORK)/data/multi30k/raw/test_2016_flickr.en \
	  --test-tgt  $(WORK)/data/multi30k/raw/test_2016_flickr.fr \
	  --src-vocab-size 10000 --tgt-vocab-size 10000 \
	  --max-ratio inf --max-len 100
	$(PY) scripts/prepare_data.py --out $(WORK)/data/wmt14/prepared_nofilter \
	  --train-src $(WORK)/data/wmt14/raw/nc9.en $(WORK)/data/wmt14/raw/europarl.sub.en \
	  --train-tgt $(WORK)/data/wmt14/raw/nc9.fr $(WORK)/data/wmt14/raw/europarl.sub.fr \
	  --dev-src   $(WORK)/data/wmt14/raw/newstest2013.en --dev-tgt $(WORK)/data/wmt14/raw/newstest2013.fr \
	  --test-src  $(WORK)/data/wmt14/raw/newstest2014.en --test-tgt $(WORK)/data/wmt14/raw/newstest2014.fr \
	  --src-vocab-size 32000 --tgt-vocab-size 32000 --max-train-pairs 500000 \
	  --max-ratio inf --max-len 100

benchmark:          ## measure tgt-words/s to size the WMT run (both arms)
	$(PY) scripts/benchmark.py --arms base attn --max-len 100 --mean-len 32 \
	  --mean-tgt-len 26 --loss-chunk 8

smoke:              ## Tier 0 grid: {fwd,rev} x 3 seeds on Multi30k
	./scripts/run_grid.sh configs/multi30k.yaml 1 2 3

wmt:                ## Tier 1: the reported Table 1 rows (long; run overnight)
	./scripts/run_grid.sh configs/wmt14_small.yaml 1

endterm-multi30k:   ## Tier 0 end-term grid: {fwd,rev} x {base,attn} x 3 seeds (~1.2h)
	DIRS="rev fwd" ARMS="base attn" ./scripts/run_grid.sh configs/multi30k_endterm.yaml 1 2 3

endterm-wmt:        ## Tier 1 end-term: reversed, both arms (~10.5h; run overnight)
	DIRS="rev" ARMS="base attn" ./scripts/run_grid.sh configs/wmt14_endterm.yaml 1

ENDTERM_M30_NOTE = Multi30k (29k image captions), 2x256. Prepared with NO length-mismatch filtering. Pipeline validation plus the seed-controlled {forward,reversed} x {fixed vector, attention} grid - NOT the paper's corpus or test set.
ENDTERM_WMT_NOTE = 0.5M pairs vs the paper's 12M; 2x512 vs 4x1000; 32k/32k vocab vs 160k/80k. Prepared with NO length-mismatch filtering and a 100-token cap. Absolute BLEU is NOT comparable to Table 1 - the direction and shape of the effects are.

results-endterm-multi30k: ## decode + score both Tier 0 arms, then join them
	for arm in "" attn; do \
	  $(PY) scripts/make_results.py --name multi30k_endterm$${arm:+_$$arm} \
	    --data-dir $(WORK)/data/multi30k/prepared_nofilter \
	    --runs-root $(WORK)/runs/multi30k_endterm \
	    --out results/multi30k_endterm/$${arm:-base} \
	    --beams 1 2 12 --seeds 1 2 3 --ensemble --per-seed \
	    --buckets endterm --max-decode-len 150 $${arm:+--arm $$arm} \
	    --scale-note "$(ENDTERM_M30_NOTE)"; \
	done
	$(PY) scripts/make_endterm_results.py --name multi30k_endterm \
	  --base results/multi30k_endterm/base --attn results/multi30k_endterm/attn \
	  --out results/multi30k_endterm --scale-note "$(ENDTERM_M30_NOTE)"

results-endterm-wmt: ## decode + score both Tier 1 arms, then join them
	for arm in "" attn; do \
	  $(PY) scripts/make_results.py --name wmt14_endterm$${arm:+_$$arm} \
	    --data-dir $(WORK)/data/wmt14/prepared_nofilter \
	    --runs-root $(WORK)/runs/wmt14_endterm \
	    --out results/wmt14_endterm/$${arm:-base} \
	    --beams 1 2 12 --seeds 1 --directions rev \
	    --buckets endterm --max-decode-len 150 $${arm:+--arm $$arm} \
	    --scale-note "$(ENDTERM_WMT_NOTE)"; \
	done
	$(PY) scripts/make_endterm_results.py --name wmt14_endterm \
	  --base results/wmt14_endterm/base --attn results/wmt14_endterm/attn \
	  --out results/wmt14_endterm --scale-note "$(ENDTERM_WMT_NOTE)"

alignments:         ## attention heatmaps (Bahdanau Fig. 3 analogue)
	PYTHONPATH=src $(PY) scripts/plot_alignments.py \
	  --checkpoint $(WORK)/runs/wmt14_endterm/rev_attn_seed1/best.pt \
	  --data-dir $(WORK)/data/wmt14/prepared_nofilter \
	  --reverse-source --out results/wmt14_endterm/alignments

probe:              ## per-length perplexity + sentence-vector saturation
	PYTHONPATH=src $(PY) scripts/probe_bottleneck.py \
	  --base-checkpoint $(WORK)/runs/wmt14_endterm/rev_seed1/best.pt \
	  --attn-checkpoint $(WORK)/runs/wmt14_endterm/rev_attn_seed1/best.pt \
	  --data-dir $(WORK)/data/wmt14/prepared_nofilter \
	  --reverse-source --out results/wmt14_endterm/probe

endterm-report:     ## rebuild the end-term report (.md and .pdf) from the results
	$(PY) scripts/make_endterm_report.py
	$(PY) scripts/md_to_pdf.py report/EE5180_endterm_report.md

endterm-slides:     ## rebuild the end-term slide deck from the results
	node scripts/make_endterm_slides.js

endterm-figures:    ## everything that does not need a training run
	$(MAKE) copy-task
	$(MAKE) alignments
	$(MAKE) probe

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

submit-endterm:     ## package the end-term deliverables
	$(MAKE) endterm-report
	$(MAKE) endterm-slides
	rm -f EE5180_seq2seq_endterm.zip
	zip -qr EE5180_seq2seq_endterm.zip . \
	  -x '*.git/*' '*__pycache__/*' '*.pytest_cache/*' '*.DS_Store' '*.pt' \
	     'EE5180_seq2seq_submission.zip' 'EE5180_seq2seq_endterm.zip' \
	     'EE5180_seq2seq_submission/*'
	@echo "wrote EE5180_seq2seq_endterm.zip ($$(du -h EE5180_seq2seq_endterm.zip | cut -f1))"

clean-runs:
	rm -rf $(WORK)/runs
