# ai-chip-and-compiler

Minimal GPU simulator and MLIR-based compiler targeting ordinary CNN/object-detection inference and small Transformer/LLM workloads.

The project is under active development. Current status:

- Minimal SIMT GPU simulator in `hw/`
- Unified Model IR in `compiler/model_ir.py`
- Manual CNN/Transformer graph frontend in `compiler/frontends/`
- MLIR text builder and `mgpu` dialect in `compiler/mlir/`
- Unit, operator end-to-end, and MLIR lit tests in `tests/`

## Setup

```bash
make setup
```

Run all test cases:

```bash
make test
```

Run individual suites:

```bash
make unit
make lit
make pytest
```

Run `make help` for formatting, pre-commit, and cleanup targets.
