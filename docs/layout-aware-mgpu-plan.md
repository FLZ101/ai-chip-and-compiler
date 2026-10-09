# Layout-Aware `mgpu-opt` Plan

## Goal

Remove the Python `compiler/codegen` fallback by making the C++ `mgpu-opt`
backend consume real buffer addresses instead of assuming dense zero-based
offsets.

After this work:

- `mgpu-opt` emits ISA for the actual Python memory layout.
- single-node and multi-node graphs lower through C++.
- Python no longer generates ISA on the canonical path.

## Problem

Current C++ emitters hardcode layouts such as:

```text
A = 0
B = sizeof(A)
C = sizeof(A) + sizeof(B)
```

This works only when the harness packs one node densely at address zero.
Real multi-node graphs allocate intermediate tensors across the whole model,
so those assumptions are wrong for subgraphs.

## Approach

Add explicit layout metadata to the MLIR module and make every C++ emitter use
that metadata through a shared `BufferMap`.

## Target representation

Add an `mgpu.memory_layout` operation:

```mlir
mgpu.memory_layout
  %arg0, %v0, %v1
  offsets = [0, 128, 256]
```

The operation associates each memref SSA value with a byte offset.

The Python driver inserts this operation after buffer allocation is known but
before `mgpu-opt` runs.

## C++ interface

Add:

```cpp
class BufferMap {
  DenseMap<Value, int64_t> offsets;
public:
  static BufferMap build(FuncOp func);
  std::optional<int64_t> offset(Value value) const;
  int64_t base(Value value) const;
};
```

`MGpuToIsaPass` builds one `BufferMap` per function and passes it to each
`emit*Isa` function.

Every hardcoded base calculation becomes:

```cpp
int64_t imageBase = bufferMap.base(imageValue);
int64_t weightBase = bufferMap.base(weightValue);
int64_t outputBase = bufferMap.base(outputValue);
```

## Python metadata injection

Current flow:

```text
graph -> MLIRBuilder -> mgpu-opt
```

New flow:

```text
graph + MemoryLayout
      -> MLIRBuilder
      -> insert mgpu.memory_layout
      -> mgpu-opt
```

The injection point belongs near `compiler/mlir/mgpu_runner.py`, where the
actual `layout` dictionary is already available.

## Work plan

### Step 1: Dialect operation

Add `MGPU_MemoryLayoutOp` to `mlir/mgpu/include/mgpu/MGPUOps.td`.

Deliverables:

- variadic memref operands
- `DenseI64ArrayAttr` byte offsets
- parser/printer round-trip

Tests:

- parse/print `mgpu.memory_layout`
- reject mismatched operand/offset counts

### Step 2: C++ `BufferMap`

Add `BufferMap` in `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`.

Deliverables:

- build from `mgpu.memory_layout`
- lookup by MLIR `Value`
- clear diagnostic when a value is missing

Tests:

- unit-style lit through `mgpu-opt`
- missing-offset error test

### Step 3: Convert common emitters

Update:

- vector add
- generic elementwise
- reduction
- matmul

Exit criteria:

- common-kernel E2E tests use real layout addresses through C++

### Step 4: Convert CNN emitters

Update:

- conv2d
- depthwise conv2d
- pooling
- global avg pool

Exit criteria:

- CNN E2E tests use real layout addresses through C++

### Step 5: Convert attention and sequence emitters

Update:

- rope
- causal mask
- kv-cache append
- kv-cache gather
- attention

Exit criteria:

- attention/sequence E2E tests use real layout addresses through C++

### Step 6: Convert transformer normalization emitters

Update:

- softmax
- layer norm
- RMS norm

Exit criteria:

- transformer norm E2E tests use real layout addresses through C++

### Step 7: Convert shape/layout emitters

Update:

- reshape
- transpose
- concat
- slice
- pad
- resize

Exit criteria:

- shape E2E tests use real layout addresses through C++

### Step 8: Convert norm/detection emitters

Update:

- batch norm
- group norm
- box decode
- NMS

Exit criteria:

- norm/detection E2E tests use real layout addresses through C++

### Step 9: Wire Python metadata injection

Update `compiler/mlir/mgpu_runner.py`:

- accept `layout`
- inject `mgpu.memory_layout` into MLIR text
- call `mgpu-opt`

Update `compiler/codegen/dispatcher.py`:

- call C++ backend for all supported single-node ops
- remove Python fallback from the canonical path

Update `tests/operator_e2e/helpers.py`:

- pass real layout to the C++ backend
- stop calling `lower_kernel_python`

### Step 10: Remove Python codegen

Delete or move:

- `compiler/codegen/activation_kernels.py`
- `compiler/codegen/attention_kernels.py`
- `compiler/codegen/cnn_kernels.py`
- `compiler/codegen/common_kernels.py`
- `compiler/codegen/norm_detection_kernels.py`
- `compiler/codegen/shape_kernels.py`
- `compiler/codegen/transformer_kernels.py`
- `compiler/codegen/dispatcher.py` Python fallback

Exit criteria:

- full test suite passes without `compiler/codegen`

## Testing strategy

Keep all existing golden E2E tests.

Add:

- `mgpu.memory_layout` round-trip lit
- missing-offset diagnostic
- one multi-node C++ lowering lit/E2E
- one previously failing multi-node model E2E

## Risks

- Many emitters currently share implicit dense-offset assumptions.
- Injecting layout text into MLIR must preserve deterministic SSA/order.
- Some operators still rely on Python MLIRBuilder decompositions that need
  target-op emission before C++ can own them.
- `Value`-based buffer lookup must handle aliases/collapse/shape views.

## Definition of done

- Every C++ `emit*Isa` function uses `BufferMap`.
- Every current E2E test passes with C++ as the only ISA backend.
- `compiler/codegen/*` no longer exists or is not imported by the canonical
  path.
- `mgpu-opt` consumes real Python buffer addresses.
