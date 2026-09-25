# Dev entry points. First time in a clone: make setup hooks
PYTHON ?= python3.12
PY := .venv/bin/python
DATASET = $(shell $(PY) -c 'from entity_resolution.config import DATASET; print(DATASET)')
OFFICIAL_VALIDATOR := dataset/student_resource/utils/validate_submission.py

.PHONY: setup hooks cache lint test score validate experiment nb public

setup:  ## .venv with pinned runtime deps + editable package + dev tools
	$(PYTHON) -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt -e ".[dev]"

hooks:  ## enable the versioned git hooks (.githooks/) for this clone
	git config core.hooksPath .githooks

cache:  ## parse every TSV once into <dataset>/.cache/*.parquet (later loads take seconds)
	$(PY) -m entity_resolution.data

lint:
	.venv/bin/ruff check src tests

test:
	.venv/bin/pytest

experiment:  ## next experiments/vNNN_<slug>/ from the template: make experiment NAME=name_tfidf
	$(PY) -m entity_resolution.tracking new $(NAME)

nb:  ## run a notebook headless, outputs saved in place: make nb NB=experiments/v001_x/v001_x.ipynb
	.venv/bin/jupyter nbconvert --to notebook --execute --inplace $(NB)

public:  ## record a leaderboard score in experiments.csv: make public V=v004 SCORE=0.8312
	$(PY) -m entity_resolution.tracking public $(V) $(SCORE)

score:  ## macro F0.5 breakdown: make score PRED=<matching.tsv> TRUTH=<ground_truth.tsv>
	$(PY) -m entity_resolution.metrics --pred $(PRED) --truth $(TRUTH)

validate:  ## output/*.tsv vs submission rules: our checker, then the organisers' validator
	$(PY) -m entity_resolution.submission --output-dir output
	$(PY) $(OFFICIAL_VALIDATOR) --matching output/matching_results.tsv \
		--candidate output/candidate_pairs.tsv --test-dir $(DATASET)/test
