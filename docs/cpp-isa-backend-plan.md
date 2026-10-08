# C++ ISA Backend Plan: `mgpu-opt` MLIR Path

## Goal

Make `mgpu-opt` the canonical C++ backend for every operator.

Python remains responsible for:

- graph construction and validation,
- PyTorch/ONNX import,
- MLIR text generation/canonicalization,
- buffer placement and test orchestration,
- invoking `mgpu-opt` and running the simulator.

C++ owns:

- all target-specific MLIR passes,
- linalg/SCF/arith/memref lowering,
- `mgpu` device lowering,
- register/address generation,
- ISA text emission.

The Python `compiler/codegen/*` ISA emitters become fallback-only and are
eventually removed.

## Current state

`mgpu-opt` already has:

- an out-of-tree `mgpu` dialect,
- `tile-linalg`,
- `map-to-mgpu`,
- `lower-to-mgpu`,
- `mgpu-to-isa`,
- specialized C++ emitters for vector add, matmul, conv2d, and
  `mgpu.attention`.

The Python operator E2E tests still call Python `compiler/codegen` directly,
so they do not currently exercise the C++ path.

## Architecture

Use the real MLIR pass pipeline:

```text
PyTorch FX / ONNX
        |
        v
Python importer/canonicalizer
        |
        v
MLIR: linalg/tosa/memref + mgpu.launch
        |
        v
mgpu-opt
  tile-linalg
  one-shot-bufferize
  map-to-mgpu
  lower-to-mgpu
  mgpu-to-isa
        |
        v
mgpu.isa op
        |
        v
simulator ISA runner
```

Where upstream MLIR has a suitable pass, use it.  Where the op is
target-specific or currently only a placeholder, implement a C++ `mgpu-opt`
pass or extend `mgpu-to-isa`.

## Target interface

Python invokes `mgpu-opt` with the deterministic pass pipeline and extracts:

- `code`
- `grid`
- `block`

from the emitted `mgpu.isa` op.

The same Python E2E harness remains the golden test oracle.

## Operator groups

### Phase 0: Connect the canonical pipeline

Status: **done**

Deliverables:

- Python `run_mgpu_pipeline(mlir_text)` helper,
- deterministic `mgpu-opt` pass pipeline,
- parser for `mgpu.isa`,
- vector-add E2E routed through `mgpu-opt`.

Files:

- `compiler/mlir/pipeline.py`
- `compiler/mlir/mgpu_runner.py`
- `tests/operator_e2e/test_mgpu_opt_pipeline_e2e.py`

Exit criteria:

- a Python E2E test emits ISA only from `mgpu-opt`.

### Phase 1: Generic linalg/SCF backend

Status: **done**

Deliverables:

- C++ `emitLinalgGenericIsa` support for rank-1 binary elementwise ops,
- C++ rank-1 reduction support for sum/mean/max,
- MLIRBuilder fixes so reduction MLIR is actually parseable,
- Python E2E and lit tests that run these ops through `mgpu-opt`.

This provides elementwise and reduction kernels through C++ without the Python
ISA emitters.

Exit criteria:

- elementwise and reduction E2E tests pass through `mgpu-opt`.

### Phase 2: Activations

Status: **done**

Use linalg/arith decomposition for:

- relu,
- leaky relu,
- silu,
- gelu,
- sigmoid.

Exit criteria:

- activation E2E tests pass through `mgpu-opt`.

### Phase 3: CNN

Status: **done**

Extend C++ lowerings for:

- conv2d,
- depthwise conv2d,
- max pool2d,
- avg pool2d,
- global avg pool.

Conv2d already has `emitConvIsa`; add depthwise/pool/global-average and wire
the linalg op recognition.

Exit criteria:

- CNN E2E tests pass through `mgpu-opt`.

### Phase 4: Attention and sequence

Status: **done**

Implement/recognize decomposition for:

- rope,
- causal mask,
- scaled dot-product attention,
- masked attention,
- kv-cache append,
- kv-cache gather.

`mgpu.attention` already exists; the remaining work is decomposition from
imported/linalg form and sequence helpers.

Exit criteria:

- attention/sequence E2E tests pass through `mgpu-opt`.

### Phase 5: Transformer normalization

Status: **done**

Lower:

- softmax,
- layer norm,
- RMS norm.

Use linalg decompositions plus the generic SCF/arith backend where possible.

Exit criteria:

- transformer normalization E2E tests pass through `mgpu-opt`.

### Phase 6: Shape and layout

Status: **done**

Lower:

- reshape,
- transpose,
- concat,
- slice,
- pad,
- resize.

Exit criteria:

- shape/layout E2E tests pass through `mgpu-opt`.

### Phase 7: Normalization and detection

Lower:

- batch norm,
- group norm,
- box decode,
- NMS.

Exit criteria:

- norm/detection E2E tests pass through `mgpu-opt`.

### Phase 8: Cut over and cleanup

Deliverables:

- Python `lower_kernel` calls `mgpu-opt`,
- Python `compiler/codegen/*` emitters are frozen or removed,
- all existing E2E tests pass through C++,
- update `docs/steps.md` and `docs/next-stage-plan.md`.

Exit criteria:

- no Python-generated ISA remains on the canonical path.

## Pass ownership

Prefer these C++ components:

- `tile-linalg`: target tile sizes.
- `map-to-mgpu`: grid/block/thread mapping.
- `lower-to-mgpu`: shared memory, barriers, async copies, `mgpu.mma`.
- `mgpu-to-isa`: final ISA emission.

New operators should be decomposed into linalg/SCF before reaching
`mgpu-to-isa`.  Specialized `emit*Isa` functions are acceptable only for
target instructions such as tensor-core MMA or ops without a clean linalg
form.

## Testing strategy

Keep every existing golden E2E test.  Each phase is complete only when the
existing test for that operator group passes with `mgpu-opt` as the backend.

Add lit tests for each C++ pass:

- parse/verify,
- linalg-to-mgpu,
- mgpu-to-isa,
- emitted ISA contains expected opcodes.

## Risks

- The current Python MLIR builder emits placeholders for some shape/norm/detection
  ops; these must be replaced with real linalg decompositions before C++ can
  lower them.
- `mgpu-to-isa` must grow from specialized emitters into a general backend.
- Upstream pass versions and custom pass order must remain deterministic.
- Debugging C++ lowerings through both lit and Python E2E requires clear
  diagnostics.

## Definition of done

- Every current operator E2E test runs through `mgpu-opt`.
- Every operator has a C++ lowering or a real linalg decomposition.
- Python no longer emits ISA on the canonical path.
- `mgpu-opt` is built by the existing build flow.
- Docs reflect that `mgpu-opt` is the only ISA backend.
