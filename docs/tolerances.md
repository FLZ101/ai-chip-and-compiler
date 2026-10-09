# Numerical Tolerances

This document records the numerical gates used by the ResNet18 execution
tests.  The simulator uses IEEE-754 FP32 arithmetic, but the compiler may
reorder reductions and use approximations such as `RSQRT` and `EXP2`.  Tolerances
are intentionally a little looser than machine epsilon to absorb those
differences without hiding real bugs.

## Cropped pretrained stem

Path:

```text
conv7x7 -> BatchNorm2d -> ReLU
```

- input: `1x3x4x4`
- reference: PyTorch on the same pretrained weights
- tolerance: `atol = 1e-4`
- profile gate: `cycles > 0`, `issued_instructions > 0`

This is the default CI smoke test.  It is intentionally cropped because the
Python functional simulator is too slow to run the full `224x224` model on
every test run.

## Full ResNet18 forward

Path:

```text
image -> stem -> 4 residual stages -> global average pool -> flatten -> Linear
```

- input: `1x3x224x224`
- reference: PyTorch `resnet18(weights=ResNet18_Weights.DEFAULT)`
- tolerance: `atol = 5e-3`, `rtol = 5e-3`
- profile gate: `cycles > 0`, profile snapshot is present
- opt-in variable: `MGPU_FULL_RESNET18=1`

Run this only when performance is not a concern:

```bash
MGPU_FULL_RESNET18=1 python -m unittest \
  tests.model_e2e.test_real_cnn_execution.TestRealCNNExecution.test_full_resnet18_forward_runs_in_simulator
```

The full test is skipped in normal `unittest discover` runs.
