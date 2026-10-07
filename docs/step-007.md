# Step 007: Shared Operator E2E Harness

This document describes the shared end-to-end test harness introduced for every
operator.

Step 007 does not add new numerical kernels.  It defines the one pipeline that
all future operator tests reuse:

```text
model graph -> MLIR -> ISA -> simulator -> device output -> NumPy comparison
```

---

## 1. Harness

File: [../tests/operator_e2e/helpers.py](../tests/operator_e2e/helpers.py)

Core helpers:

- `allocate_global_buffers`: places host tensors into simulated global memory.
- `read_tensor`: reads a placed tensor back as a NumPy array.
- `compare_against_numpy`: compares simulator outputs with golden tensors.
- `run_operator_e2e`: builds MLIR, invokes a pluggable lowering, runs the
  simulator, reads outputs, and compares them.

The MLIR-to-ISA stage is passed as a `LoweringFn`:

```python
LoweringFn = Callable[
    [Graph, str, MemoryLayout],
    tuple[str, tuple[tuple[int, ...], tuple[int, ...]]],
]
```

This keeps the harness reusable while individual kernel implementations arrive
in later steps.

---

## 2. Reference vector-add lowering

The harness includes `vector_add_lowering`, which handles a single rank-1
elementwise add graph.  It proves the full pipeline works end to end and serves
as the pattern for the common kernels added in Step 008.

---

## 3. Files changed

- [../tests/operator_e2e/helpers.py](../tests/operator_e2e/helpers.py)
- [../tests/operator_e2e/test_shared_harness_e2e.py](../tests/operator_e2e/test_shared_harness_e2e.py)

---

## 4. Tests

The new test verifies:

- graph construction,
- deterministic MLIR text,
- ISA generation,
- simulator execution,
- output readback,
- NumPy comparison,
- profile collection.

Run:

```bash
source .venv/bin/activate
python -m unittest discover -v
```

All existing tests continue to pass.
