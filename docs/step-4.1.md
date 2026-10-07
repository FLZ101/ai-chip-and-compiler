# Step 4.1: Execute MMA in the Simulator

This document describes the Phase 4.1 implementation.

The simulator now recognizes an `MMA` instruction. In the current MVP, `MMA`
models one tensor-core fragment element as:

```text
dst = a * b + c
```

This keeps the execution model scalar while establishing the instruction and
test path that later fragment-layout work will extend.

---

## 1. What was implemented

- Added `MMA` to the ISA parser.
- Added `MMA` to the simulator ALU.
- Added parser coverage in
  [../tests/unit/test_isa.py](../tests/unit/test_isa.py).
- Added an end-to-end 2x2 tile test in
  [../tests/operator_e2e/test_mma_e2e.py](../tests/operator_e2e/test_mma_e2e.py).

The E2E test loads two 2x2 matrices, computes each output element with a
sequence of `MMA` instructions, stores the result, and compares it with
NumPy's `A @ B`.

---

## 2. Scope

This phase intentionally implements functional MMA only. Fragment layout,
shared-memory swizzling, and true warp-wide tensor-core execution are deferred
to Phase 4.2.

---

## 3. Files changed

- [../hw/isa.py](../hw/isa.py)
- [../hw/core.py](../hw/core.py)
- [../tests/unit/test_isa.py](../tests/unit/test_isa.py)
- [../tests/operator_e2e/test_mma_e2e.py](../tests/operator_e2e/test_mma_e2e.py)
