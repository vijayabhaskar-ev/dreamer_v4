# One command per check of the paper's results. None of these trains a model.
PY ?= python

.PHONY: check figures tests all
check:            ## recompute 55 numbers of the paper from evaluation/ and results_corrected/, compare with the paper
	$(PY) analysis/recompute_from_results.py
figures:          ## redraw Figures 1 and 2 of the paper into analysis/figures/
	cd analysis/figures && $(PY) make_fig_beforeafter.py && $(PY) make_fig_stepsize.py
tests:            ## unit tests of the imagination algebra, the rollout contract and the attention firewall
	$(PY) -m tests.test_imagination_algorithms && $(PY) -m tests.test_imagination_rollout && $(PY) -m tests.test_gradient_isolation
all: check figures tests
