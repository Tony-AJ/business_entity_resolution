# Dev entry points. First time in a clone: make setup hooks
PYTHON ?= python3.12
PY := .venv/bin/python

.PHONY: setup hooks lint test

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
