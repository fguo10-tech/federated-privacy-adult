# federated-privacy-adult — common tasks. Run from the repository root.
#
#   make install        editable install with every optional extra
#   make test           full test suite (no network)
#   make smoke          tiny end-to-end run of the benchmark on synthetic toy data
#   make data           download UCI Adult from the canonical UCI source
#   make federated      the benchmark on UCI Adult (20 rounds x 5 seeds, DP sweep)
#   make results        rebuild every table and chart from the reference run and sync the docs
#
# Outputs of your own runs go to experiments/results/runs/ (git-ignored).
# experiments/results/reference/ holds the run behind every [Reproduced] number in the docs;
# only `make reference` re-creates its raw results (`make results` rebuilds the tables and
# charts derived from them, under reference/generated and reference/before_after).

PYTHON ?= python
WORKERS ?= 4
RUNS := experiments/results/runs
REFERENCE := experiments/results/reference
CODE := src tests experiments examples data

.DEFAULT_GOAL := help
.PHONY: help install lint format test test-fast data smoke federated federated-toy reference \
        results docs-check notebooks clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-22s %s\n", $$1, $$2}'

install: ## Editable install with every optional extra (CPU is enough)
	$(PYTHON) -m pip install -e ".[all]"

lint: ## Static checks: ruff and black --check
	$(PYTHON) -m ruff check $(CODE)
	$(PYTHON) -m black --check $(CODE)

format: ## Auto-fix lint findings and format the code
	$(PYTHON) -m ruff check --fix $(CODE)
	$(PYTHON) -m black $(CODE)

test: ## Full test suite (no network; Adult-only tests skip if data/adult is absent)
	$(PYTHON) -m pytest

test-fast: ## Test suite without the slower smoke runs
	$(PYTHON) -m pytest -m "not slow"

data: ## Download UCI Adult from archive.ics.uci.edu into data/adult (git-ignored) and verify checksums
	$(PYTHON) data/download_adult.py

smoke: ## Tiny end-to-end run of the benchmark on synthetic toy data; under a minute, no download
	$(PYTHON) -m experiments.run_federated_benchmark --config configs/federated_fedavg.yaml \
		--suite all --smoke --out $(RUNS)/smoke_federated
	$(PYTHON) -m experiments.plot_federated --run-dir $(RUNS)/smoke_federated

federated: data ## The benchmark: UCI Adult, workclass clients + IID control, 20 rounds x 5 seeds, DP sweep
	$(PYTHON) -m experiments.run_federated_benchmark --config configs/federated_adult.yaml \
		--suite main --workers $(WORKERS) --out $(RUNS)/federated_adult
	$(PYTHON) -m experiments.plot_federated --run-dir $(RUNS)/federated_adult

federated-toy: ## The benchmark on the synthetic toy domains (no download)
	$(PYTHON) -m experiments.run_federated_benchmark --config configs/federated_fedavg.yaml \
		--suite all --workers $(WORKERS) --out $(RUNS)/federated_toy
	$(PYTHON) -m experiments.plot_federated --run-dir $(RUNS)/federated_toy

reference: data ## Re-create the reference run that the documentation cites (overwrites reference/)
	$(PYTHON) -m experiments.run_reference --workers $(WORKERS)
	$(MAKE) results

results: ## Rebuild tables, Before/After CSVs and charts from reference/, then sync them into the docs
	$(PYTHON) -m experiments.build_results
	$(PYTHON) -m experiments.sync_docs

docs-check: ## Fail if any generated table in the documentation is out of date
	$(PYTHON) -m experiments.sync_docs --check

notebooks: ## Execute the notebook top to bottom into notebooks/_executed (git-ignored)
	$(PYTHON) -m experiments.run_notebooks

clean: ## Remove caches and your own run outputs (never the reference run, never downloaded data)
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info $(RUNS) notebooks/_executed
	find . -name __pycache__ -type d -not -path "./.venv/*" -prune -exec rm -rf {} +
