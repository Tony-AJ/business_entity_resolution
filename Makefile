# Dev entry points. First time in a clone: make setup hooks
PYTHON ?= python3.12
PY := .venv/bin/python
DATASET = $(shell $(PY) -c 'from entity_resolution.config import DATASET; print(DATASET)')
OFFICIAL_VALIDATOR := dataset/student_resource/utils/validate_submission.py

.PHONY: setup hooks lint test score validate

setup:  ## .venv with pinned runtime deps + editable package + dev tools
	$(PYTHON) -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt -e ".[dev]"

hooks:  ## enable the versioned git hooks (.githooks/) for this clone
	git config core.hooksPath .githooks

lint:
	.venv/bin/ruff check src tests

test:
	.venv/bin/pytest

score:  ## macro F0.5 breakdown: make score PRED=<matching.tsv> TRUTH=<ground_truth.tsv>
	$(PY) -m entity_resolution.metrics --pred $(PRED) --truth $(TRUTH)

validate:  ## output/*.tsv vs submission rules: our checker, then the organisers' validator
	$(PY) -m entity_resolution.submission --output-dir output
	$(PY) $(OFFICIAL_VALIDATOR) --matching output/matching_results.tsv \
		--candidate output/candidate_pairs.tsv --test-dir $(DATASET)/test
