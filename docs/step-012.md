# Step 012: Transformer Normalization and Softmax

This document describes the stable numeric primitives used by attention.

Step 012 adds:

- online `Softmax`,
- `LayerNorm`,
- `RMSNorm`.

Each operator is tested through the shared E2E harness against a NumPy golden
reference.

---

## 1. Transformer lowering

File: [../compiler/codegen/transformer_kernels.py](../compiler/codegen/transformer_kernels.py)

The initial Transformer lowerings support rank-2 tensors shaped `(rows, dim)`,
with one thread per row and unrolled reduction over `dim`.

### Softmax

The softmax kernel uses the stable three-pass form:

1. compute the row maximum,
2. sum `exp(x - max)`,
3. write `exp(x - max) / sum`.

The simulator's `EXP2` and `RCP` instructions implement the exponential and
division.

### LayerNorm

For each row:

```text
mean = sum(x) / dim
var  = sum(x^2) / dim - mean^2
y    = (x - mean) * rsqrt(var + eps) * weight + bias
```

### RMSNorm

For each row:

```text
rms = sqrt(mean(x^2) + eps)
y   = x / rms * weight
```

---

## 2. MLIR builder update

[../compiler/mlir/mlir_builder.py](../compiler/mlir/mlir_builder.py) now emits
placeholder `linalg.generic` text for softmax, layer norm, and RMS norm.  The
executable path remains the ISA lowering in this step.

---

## 3. Files changed

- [../compiler/codegen/transformer_kernels.py](../compiler/codegen/transformer_kernels.py)
- [../compiler/codegen/dispatcher.py](../compiler/codegen/dispatcher.py)
- [../compiler/mlir/mlir_builder.py](../compiler/mlir/mlir_builder.py)
- [../tests/operator_e2e/test_transformer_norm_softmax_e2e.py](../tests/operator_e2e/test_transformer_norm_softmax_e2e.py)

---

## 4. Tests

New tests cover:

- softmax, including a numerically stressful large-input row,
- layer norm,
- RMS norm.

Run:

```bash
source .venv/bin/activate
python -m unittest discover -v
```

All existing tests continue to pass.
