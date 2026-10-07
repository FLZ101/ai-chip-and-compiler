# Numerical Tolerances

This document describes the numerical comparison policy used by the E2E test
harness.

## Default tolerances

The shared harness in
[../tests/operator_e2e/helpers.py](../tests/operator_e2e/helpers.py) defaults to:

- relative tolerance `rtol = 1e-5`
- absolute tolerance `atol = 1e-6`

Tests may override these values for operators whose reference formula is
approximate.

## Operator notes

- `GELU` uses the common `tanh` approximation in both the simulator and the
  NumPy reference, so its comparison is intentionally formula-consistent.
- Attention and softmax use `max` subtraction, `EXP2`, and `RCP`; tests allow
  `atol = 1e-5` because of repeated FP32 special-function rounding.
- CNN, normalization, and detection helpers use direct arithmetic and default
  tolerances.
- Shape/layout copy kernels are expected to match exactly, but the shared
  harness still uses the default tolerance for consistency.

## Policy

Tests must compare against deterministic NumPy references.  Silent loose
tolerances are discouraged; when a larger tolerance is required, the test
should document why near the assertion.
