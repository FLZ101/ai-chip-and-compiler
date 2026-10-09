PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin
PIP := $(BIN)/pip
PY := $(BIN)/python
SETUP_STAMP := $(VENV)/.make-setup

.DEFAULT_GOAL := help

.PHONY: help setup unit lit test pytest format precommit mgpu-opt cpp-sim cpp-sim-test benchmark-cpp-sim full-resnet18 full-resnet18-cpp full-resnet18-python clean clean-venv

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
	@printf "  make cpp-sim      Build the native C++ simulator\n"
	@printf "  make cpp-sim-test Build and run native C++ simulator tests\n"
	@printf "  make benchmark-cpp-sim  Compare Python and C++ simulator speed\n"
	@printf "  make full-resnet18       Run full ResNet18 with native simulator\n"
	@printf "  make full-resnet18-cpp   Alias for full-resnet18\n"
	@printf "  make full-resnet18-python  Run full ResNet18 with Python simulator\n"
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

cpp-sim:
	cmake -S cpp_sim -B .deps/cpp-sim-build \
	  -DPython3_EXECUTABLE=$(abspath $(PY))
	cmake --build .deps/cpp-sim-build -j8

cpp-sim-test: cpp-sim
	ctest --test-dir .deps/cpp-sim-build --output-on-failure

benchmark-cpp-sim: cpp-sim
	$(PY) scripts/benchmark_cpp_sim.py

FULL_RESNET18_TEST = tests.model_e2e.test_real_cnn_execution.TestRealCNNExecution.test_full_resnet18_forward_runs_in_simulator

full-resnet18: cpp-sim
	MGPU_FULL_RESNET18=1 MGPU_CPP_SIM=1 \
	  $(PY) -m unittest $(FULL_RESNET18_TEST) -v

full-resnet18-cpp: full-resnet18

full-resnet18-python: setup
	MGPU_FULL_RESNET18=1 \
	  $(PY) -m unittest $(FULL_RESNET18_TEST) -v

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -f tests/mlir/.lit_test_times.txt

clean-venv:
	rm -rf $(VENV)
