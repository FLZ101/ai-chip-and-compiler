# Step 4.4: Lower Conv and Attention to MMA

This document describes the Phase 4.4 implementation.

The simulator now executes warp-wide 16x16x16 tensor-core `MMA`.  Phase 4.4
uses that instruction for the two remaining important workloads:

* ordinary 2D convolution, lowered as im2col matrix multiplication;
* scaled-dot-product attention, lowered as two matrix multiplications with a
  softmax stage between them.

Both lowerings are currently for the single-warp MVP.  Invalid matrix rows and
columns are masked to zero, and shared memory is used as the boundary between
the tensor-core and SIMT phases.

The C++ `mgpu-to-isa` pass now contains matching target emitters as well:
`emitConvIsa` handles `linalg.conv_2d_nchw_fchw`, and `emitAttentionIsa`
handles the new `mgpu.attention` operation.

## 1. Conv2D

The new path in
[../compiler/codegen/cnn_kernels.py](../compiler/codegen/cnn_kernels.py)
treats:

```text
C[spatial, output_channel] =
    im2col(image)[spatial, reduction] @ weight_matrix[reduction, output_channel]
```

where:

```text
reduction = c * kh * kw
spatial   = oh * ow
```

The warp loads the A and B fragments directly from the original NCHW image and
weight buffers, so no materialized im2col buffer is required.  The MMA result
is stored to shared memory, synchronized, and a SIMT copy phase writes only the
valid NCHW output elements and adds bias.

The fast path is used when:

```text
spatial <= 16
reduction <= 16
output_channels <= 16
```

The existing scalar FMA lowering remains as a fallback for larger shapes.

## 2. Attention

The new path in
[../compiler/codegen/attention_kernels.py](../compiler/codegen/attention_kernels.py)
uses two MMA instructions:

```text
scores  = Q @ K^T
weights = softmax(scores * scale)
output  = weights @ V
```

The score tile is written to shared memory, softmax is executed as a SIMT
phase, and the normalized weights are read back as the A fragment for the
second MMA.  The final output tile is copied from shared memory to global
memory.

The path supports the current `S <= 16`, `D <= 16` test cases.  Masked
attention applies the mask during the softmax phase.

## 3. Tests

- [../tests/operator_e2e/test_cnn_kernels_e2e.py](../tests/operator_e2e/test_cnn_kernels_e2e.py)
  now requires `MMA` for the regular conv2d case.
- [../tests/operator_e2e/test_attention_kernels_e2e.py](../tests/operator_e2e/test_attention_kernels_e2e.py)
  requires `MMA` for scaled-dot-product and masked attention.

## 4. Files changed

- [../compiler/codegen/cnn_kernels.py](../compiler/codegen/cnn_kernels.py)
- [../compiler/codegen/attention_kernels.py](../compiler/codegen/attention_kernels.py)
- [../mlir/mgpu/include/mgpu/MGPUOps.td](../mlir/mgpu/include/mgpu/MGPUOps.td)
- [../mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp](../mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp)
- [../tests/operator_e2e/test_cnn_kernels_e2e.py](../tests/operator_e2e/test_cnn_kernels_e2e.py)
- [../tests/operator_e2e/test_attention_kernels_e2e.py](../tests/operator_e2e/test_attention_kernels_e2e.py)
- [../tests/mlir/mgpu_to_isa_conv.mlir](../tests/mlir/mgpu_to_isa_conv.mlir)
- [../tests/mlir/mgpu_to_isa_attention.mlir](../tests/mlir/mgpu_to_isa_attention.mlir)
