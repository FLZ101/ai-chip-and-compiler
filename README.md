# ai-chip-and-compiler

Minimal GPU simulator and MLIR-based compiler targeting ordinary CNN/object-detection inference and small Transformer/LLM workloads.

The project is under active development. Current status:

- Minimal ISA parser/printer in `hw/isa.py`
- Byte-addressable memory model in `hw/memory.py`
- SIMT core and device launcher in `hw/core.py`
- Initial ISA and vector-add tests in `tests/`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m unittest discover -v
```
