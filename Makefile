PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin
PIP := $(BIN)/pip
PY := $(BIN)/python
SETUP_STAMP := $(VENV)/.make-setup

.DEFAULT_GOAL := help

.PHONY: help setup unit lit test pytest format precommit mgpu-opt clean clean-venv

help:
	@printf "Targets:\n"
	@printf "  make setup        Create .venv and install test dependencies\n"
	@printf "  make unit         Run unittest suite\n"
	@printf "  make lit          Run MLIR lit tests\n"
	@printf "  make test         Run unittest and lit tests\n"
	@printf "  make pytest       Run pytest suite\n"
	@printf "  make format       Format Python code with black\n"
	@printf "  make precommit    Run all pre-commit hooks\n"
	@printf "  make mgpu-opt     Download/build MLIR and out-of-tree mgpu-opt\n"
	@printf "  make clean        Remove generated caches/test artifacts\n"
	@printf "  make clean-venv   Remove .venv\n"

setup: $(SETUP_STAMP)

$(SETUP_STAMP): pyproject.toml
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e '.[test]'
	$(BIN)/pre-commit install
	touch $(SETUP_STAMP)

unit: setup
	$(PY) -m unittest discover -v

lit: setup
	$(BIN)/lit -v tests/mlir

test: unit lit

pytest: setup
	$(PY) -m pytest -v

format: setup
	$(BIN)/black hw compiler tests

precommit: setup
	$(BIN)/pre-commit run --all-files

mgpu-opt:
	./scripts/build_mlir.sh

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -f tests/mlir/.lit_test_times.txt

clean-venv:
	rm -rf $(VENV)
