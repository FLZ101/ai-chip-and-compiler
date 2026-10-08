# Step 013: Attention, RoPE, and KV-cache

This document describes the first Transformer attention/sequence kernels.

Step 013 adds:

- `RoPE`,
- causal mask generation,
- scaled dot-product attention,
- masked attention,
- `KVCacheAppend`,
- `KVCacheGather`.

All operators are tested through the shared E2E harness with NumPy golden
references.

---

## 1. Attention lowering

File: [../compiler/codegen/attention_kernels.py](../compiler/codegen/attention_kernels.py)

The initial attention path supports small deterministic shapes:

- attention: `B=1`, `H=1`, `S=2`, `D=2`,
- RoPE: rank-2 `(rows, dim)`,
- KV-cache append: equal cache/value shapes,
- KV-cache gather: `B=1`, `H=1`.

The initial attention path uses a stable three-pass form:

1. compute max score,
2. sum `exp(score - max)`,
3. accumulate output with softmax weights.

Masked attention applies a `-inf` mask before softmax.

Phase 4.4 adds an MMA path for `S <= 16`, `D <= 16` attention.  It computes
`Q @ K^T` and `softmax(...) @ V` as two padded tensor-core MMA operations with
a SIMT softmax phase between them.

---

## 2. Sequence helpers

- `causal_mask` emits a `(1, 1, Q, K)` lower-triangular mask.
- `RoPE` rotates adjacent dimension pairs.
- `KVCacheAppend` is currently a copy kernel for equal-shaped cache/value.
- `KVCacheGather` gathers one position from a rank-4 cache.

These restrictions are intentionally documented and can be generalized in
later steps.

---

## 3. MLIR builder update

[../compiler/mlir/mlir_builder.py](../compiler/mlir/mlir_builder.py) now emits
placeholder MLIR for attention, RoPE, causal mask, and KV-cache ops.  The
executable path remains the ISA lowering.

---

## 4. Files changed

- [../compiler/codegen/attention_kernels.py](../compiler/codegen/attention_kernels.py)
- [../compiler/codegen/dispatcher.py](../compiler/codegen/dispatcher.py)
- [../compiler/mlir/mlir_builder.py](../compiler/mlir/mlir_builder.py)
- [../tests/operator_e2e/test_attention_kernels_e2e.py](../tests/operator_e2e/test_attention_kernels_e2e.py)

---

## 5. Tests

New tests cover:

- causal mask,
- RoPE,
- scaled dot-product attention,
- masked attention,
- KV-cache append,
- KV-cache gather.

Run:

```bash
source .venv/bin/activate
python -m unittest discover -v
```

All existing tests continue to pass.
