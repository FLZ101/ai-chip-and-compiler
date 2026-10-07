# Step 011: CNN Normalization and Detection Helpers

This document describes the CNN normalization and object-detection helper
milestone.

Step 011 adds:

- inference-mode `BatchNorm`,
- `GroupNorm`,
- `BoxDecode`,
- simple greedy `NMS`.

It also enables the shared E2E harness to run multi-node graphs, so a small
detector-head graph can compile and execute.

---

## 1. Normalization kernels

File: [../compiler/codegen/norm_detection_kernels.py](../compiler/codegen/norm_detection_kernels.py)

`BatchNorm` and `GroupNorm` are implemented as one-thread-per-output-element
kernels for NCHW tensors with batch size 1.

- `BatchNorm` computes:

```text
(x - mean) * rsqrt(var + eps) * scale + bias
```

- `GroupNorm` uses unrolled group-statistics loops and branchless per-element
  selection, then normalizes each element with its group mean/variance.

---

## 2. Detection helpers

`BoxDecode` converts `[cx, cy, w, h]` boxes plus deltas into `[x1, y1, x2, y2]`:

```text
cx = box_cx + delta_cx * box_w
cy = box_cy + delta_cy * box_h
w  = box_w * exp(delta_w)
h  = box_h * exp(delta_h)
```

`NMS` is a simple greedy suppression without score sorting:

- one thread per box,
- suppress box `i` if another box `j` has a higher score and IoU above the
  threshold,
- write kept indices and `-1` for suppressed boxes.

This is intentionally simple and deterministic.

---

## 3. Multi-node E2E harness

[../tests/operator_e2e/helpers.py](../tests/operator_e2e/helpers.py) now handles
graphs with more than one node:

1. Allocate buffers for all graph inputs and intermediate node outputs.
2. Lower and launch each node in graph order.
3. Read only the declared graph outputs.

This is what allows the detector-head test to run:

```text
boxes/deltas -> BoxDecode -> NMS -> selected indices
```

---

## 4. Files changed

- [../compiler/codegen/norm_detection_kernels.py](../compiler/codegen/norm_detection_kernels.py)
- [../compiler/codegen/dispatcher.py](../compiler/codegen/dispatcher.py)
- [../compiler/mlir/mlir_builder.py](../compiler/mlir/mlir_builder.py)
- [../tests/operator_e2e/helpers.py](../tests/operator_e2e/helpers.py)
- [../tests/operator_e2e/test_norm_detection_kernels_e2e.py](../tests/operator_e2e/test_norm_detection_kernels_e2e.py)

---

## 5. Tests

New tests cover:

- `BatchNorm`,
- `GroupNorm`,
- `BoxDecode`,
- `NMS`,
- a two-node detector-head graph.

Run:

```bash
source .venv/bin/activate
python -m unittest discover -v
```

All existing tests continue to pass.
