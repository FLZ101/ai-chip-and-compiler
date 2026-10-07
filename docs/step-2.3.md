# Step 2.3: Canonicalize Imported MLIR

This document describes the implementation of Phase 2.3 from
[next-stage-steps.md](next-stage-steps.md).

The raw PyTorch FX and ONNX modules from Phases 2.1 and 2.2 are now lowered
through torch-mlir's decomposition, shape inference, canonicalization, and
backend pipelines into a common high-level dialect set.

---

## 1. What was implemented

Added [../compiler/mlir/canonicalizer.py](../compiler/mlir/canonicalizer.py):

- `MLIRCanonicalizer`
- `CanonicalizedMLIRResult`
- `MLIRCanonicalizationError`

The canonicalizer detects the source module:

- FX modules run the `torchdynamo-export-to-torch-backend-pipeline`
  preparation passes,
- ONNX modules run the `torch-onnx-to-torch-backend-pipeline` preparation
  passes,

then both paths run `torch-mlir`'s backend lowering to either:

- `linalg-on-tensors`, the default, or
- `tosa`.

---

## 2. Usage

```python
from compiler.mlir import MLIRCanonicalizer

canonical = MLIRCanonicalizer().canonicalize(imported)
print(canonical.text)
print(canonical.target)  # "linalg-on-tensors"
print(canonical.source)  # "fx" or "onnx"
```

---

## 3. Target dialects

For `linalg-on-tensors`, output uses:

- `linalg`
- `tensor`
- `arith`
- `math`

For `tosa`, output uses:

- `tosa`
- `tensor`
- `arith`
- `math`

`memref` is intentionally deferred to Phase 3.3 bufferization.

---

## 4. Connecting to `mgpu-opt`

`MLIRCanonicalizer` is the frontend lowering step. It detects whether the input
is FX or ONNX, runs the corresponding torch-mlir preparation pipeline, then
runs the shared backend lowering to `linalg-on-tensors` or `tosa`.

The simplest handoff to `mgpu-opt` is through the returned MLIR text:

```python
from pathlib import Path
import subprocess

canonical = MLIRCanonicalizer().canonicalize(imported)
Path("model.linalg.mlir").write_text(canonical.text)

subprocess.run(
    [
        ".deps/mgpu-build/bin/mgpu-opt",
        "model.linalg.mlir",
        "--pass-pipeline=builtin.module(canonicalize,cse)",
        "-o",
        "model.canonical.mlir",
    ],
    check=True,
)
```

Later the same file handoff becomes the full target pipeline:

```python
subprocess.run(
    [
        ".deps/mgpu-build/bin/mgpu-opt",
        "model.linalg.mlir",
        "--pass-pipeline=builtin.module("
        "tile-linalg,"
        "one-shot-bufferize,"
        "map-to-mgpu,"
        "lower-to-mgpu,"
        "mgpu-to-isa"
        ")",
        "-o",
        "model.isa",
    ],
    check=True,
)
```

Current boundary:

```text
torch/ONNX model
      -> Python importer
      -> Python MLIRCanonicalizer
      -> canonicalized .mlir file
      -> mgpu-opt target passes
      -> simulator ISA
```

`mgpu-opt` can already parse the canonicalized module because the standard
`linalg`, `tensor`, `arith`, and `math` dialects are available. The later
`tile-linalg`, `one-shot-bufferize`, `map-to-mgpu`, `lower-to-mgpu`, and
`mgpu-to-isa` passes are still to be implemented in the out-of-tree tool.

---

## 5. Determinism

Re-importing and canonicalizing the same model produces identical MLIR text.
This is covered by
[../tests/unit/test_mlir_canonicalizer.py](../tests/unit/test_mlir_canonicalizer.py).

---

## 6. Tests

New tests cover:

- FX import canonicalized to `linalg-on-tensors`,
- deterministic FX canonicalization,
- ONNX import canonicalized to `linalg-on-tensors`,
- ONNX import canonicalized to `tosa`.

Run:

```bash
source .venv/bin/activate
python -m unittest -v tests.unit.test_mlir_canonicalizer
```

---

## 7. Files changed

- [../compiler/mlir/canonicalizer.py](../compiler/mlir/canonicalizer.py)
- [../compiler/mlir/__init__.py](../compiler/mlir/__init__.py)
- [../tests/unit/test_mlir_canonicalizer.py](../tests/unit/test_mlir_canonicalizer.py)
