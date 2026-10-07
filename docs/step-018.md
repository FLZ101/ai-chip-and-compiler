# Step 018: Hardening and Documentation

This document describes the final hardening pass.

Step 018 does not add new operators or examples. It improves diagnostics,
boundary checks, and documentation, then verifies the complete suite.

---

## 1. Diagnostics and boundary checks

Changes:

- The shared E2E harness now validates that every graph input tensor has the
  exact shape declared by the Model IR.
- The harness checks global-memory capacity before placing tensors.
- The runtime `Driver.allocate` validates shape dimensions and checks memory
  capacity before registering a buffer.

These changes turn silent memory corruption or shape mismatch bugs into clear
errors at the host/runtime boundary.

---

## 2. Tolerance documentation

Added [tolerances.md](tolerances.md), which documents:

- default `rtol = 1e-5` and `atol = 1e-6`,
- special-function/approximation notes for `GELU`, attention, and softmax,
- the policy that larger tolerances must be documented at the test site.

---

## 3. Type documentation update

Updated [types.md](types.md) to state clearly that:

- the simulator and ISA code generator currently execute FP32,
- the Model IR already carries FP16/BF16/FP8/INT4/INT8 metadata for future
  lowering.

---

## 4. Verification

The full project now passes:

- Black via pre-commit,
- all unittest suites,
- MLIR lit tests,
- operator E2E tests,
- model E2E tests.

Run:

```bash
source .venv/bin/activate
pre-commit run --all-files
python -m unittest discover -v
lit -q tests/mlir
```

---

## 5. Files changed

- [../tests/operator_e2e/helpers.py](../tests/operator_e2e/helpers.py)
- [../runtime/driver.py](../runtime/driver.py)
- [../docs/tolerances.md](../docs/tolerances.md)
- [../docs/types.md](../docs/types.md)
