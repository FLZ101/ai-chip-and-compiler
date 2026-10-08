# Step 010: CNN Kernels

This document describes the CNN kernel milestone.

Step 010 adds ordinary CNN operators:

- `Conv2d`,
- `DepthwiseConv2d`,
- `MaxPool2d`,
- `AvgPool2d`,
- `GlobalAvgPool`.

All operators are tested through the shared E2E harness with NumPy golden
references.

---

## 1. CNN lowering

File: [../compiler/codegen/cnn_kernels.py](../compiler/codegen/cnn_kernels.py)

The initial CNN lowerings are direct, one-thread-per-output-element kernels.
They are intentionally simple and are sufficient for the small deterministic
tests used by the compiler pipeline.

Current restrictions:

- batch size `N == 1`,
- `Conv2d` supports `groups=1` and `dilation=1`,
- convolution padding is supported with branchless clamped loads,
- pooling supports `padding=0`.

The initial conv kernels use unrolled channel/kernel loops and `FMA`.  Phase
4.4 adds an im2col-style MMA path for regular `Conv2d` when the spatial,
reduction, and output-channel dimensions each fit the 16x16 tensor-core tile.

---

## 2. MLIR builder update

[../compiler/mlir/mlir_builder.py](../compiler/mlir/mlir_builder.py) now emits:

- `linalg.conv_2d_nchw_fchw` for `Conv2d`,
- `linalg.depthwise_conv_2d_nchw_chw` for `DepthwiseConv2d`,
- placeholder `linalg.generic` text for pooling ops.

The executable pooling path is the ISA lowering, not the placeholder MLIR.

---

## 3. Files changed

- [../compiler/codegen/cnn_kernels.py](../compiler/codegen/cnn_kernels.py)
- [../compiler/codegen/dispatcher.py](../compiler/codegen/dispatcher.py)
- [../compiler/mlir/mlir_builder.py](../compiler/mlir/mlir_builder.py)
- [../tests/operator_e2e/test_cnn_kernels_e2e.py](../tests/operator_e2e/test_cnn_kernels_e2e.py)

---

## 4. Tests

New tests cover:

- `Conv2d` with bias,
- `DepthwiseConv2d`,
- `MaxPool2d`,
- `AvgPool2d`,
- `GlobalAvgPool`.

Run:

```bash
source .venv/bin/activate
python -m unittest discover -v
```

All existing tests continue to pass.
