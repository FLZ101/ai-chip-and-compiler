# GPT-2-Small Full Execution Plan

This document defines the remaining work to run a real pretrained GPT-2-small
decoder in the simulator and compare greedy token IDs with PyTorch.

## 1. Goal

Run this exact path:

```text
input_ids (1 x seq)
  -> token embedding
  -> positional embedding
  -> 12 x GPT-2 transformer block
  -> final LayerNorm
  -> lm_head Linear
  -> argmax
  -> next token id
```

The first milestone is a short static sequence, e.g. `seq = 8`, with
`use_cache=False`.  KV-cache decoding is a follow-up milestone.

## 2. Current state

Already available:

- GPT-2-small checkpoint download/caching.
- torch-mlir import and canonicalization of a single GPT-2 block.
- simulator execution of pretrained LayerNorm.
- general `Linear` lowering.
- masked softmax lowering.
- typed runtime buffers and lifetime-aware memory planner.
- wave block scheduler and runtime launch queue.
- INT8/INT4/FP8 typed memory and quantized matmul support.
- native software full-model runner used by ResNet18.
- software kernels for matmul/linear, conv, batch norm, ReLU, add, pooling,
  reshape, transpose, concat, slice, resize, softmax, masked softmax,
  layer norm, RMS norm, rope, KV cache, and attention.
- software `embedding_gather` and `argmax` Model IR ops with reference tests.
- software large masked-attention reference test for `heads=12`,
  `head_dim=64`, and `seq=8`.
- software large linear reference test for `128x256 -> 128x512`.
- full manual GPT-2 Model IR graph construction with real dimensions.
- multi-head attention lowering through `MLIRBuilder`, including broadcast
  causal masks.
- GPT-2-sized software linear tests for `8x768 -> 8x3072` and
  `8x3072 -> 8x768`.
- static `tm_tensor.scan` lowering to `linalg.matmul`.
- initial `linalg.generic` lowering for integer casts, fills, compares,
  selects, and selected float/reduction patterns.
- complete software operator coverage for the full GPT-2 graph, including
  batched `Linear` and exact GELU.
- mgpu-vs-software differential tests for the supported operators.

Missing for full GPT-2:

- wire pretrained weights into the full Model IR graph and validate it against
  PyTorch,
- full static `seq=8` forward through the software operator runner,
- greedy autoregressive loop orchestration,
- numerical token/logit gates against PyTorch.

## 2.1 Execution strategy

Use a **software-first** execution strategy for full GPT-2:

```text
full GPT-2 Model IR
  -> native software operator runner (fast path)
  -> PyTorch numerical gates

mgpu / ISA path
  -> per-operator correctness oracle on cropped shapes
  -> bounded lowering and lit tests
```

The full GPT-2 graph and decode loop are executed with the existing native
software kernels plus Python/NumPy fallbacks. The `mgpu-opt`/simulator path is
used to validate each operator where shapes are supported, but full
GPT-2-sized `mgpu-to-isa` execution is not a correctness prerequisite.

## 3. Target sizes

GPT-2-small:

- vocab: `50257`
- hidden: `768`
- heads: `12`
- layers: `12`
- head dim: `64`
- max context: `1024`

At `seq = 8`, FP32:

- weights: ~500 MB
- activations: tens of MB
- KV cache later: `2 * 12 * seq * 64 * 4` bytes per token step

## 4. Phases

### Phase A: Import a static full model

Status: **done**

Implement:

- export/import full `GPT2LMHeadModel` with
  `example_kwargs={"use_cache": False}`,
- canonicalize to `linalg-on-tensors`,
- verify the module,
- freeze input to `1 x seq`.

Files:

- `compiler/mlir/importer.py`
- `compiler/mlir/canonicalizer.py`
- `tests/model_e2e/test_real_pretrained_llm.py`

Exit criteria:

- full GPT-2-small canonicalizes without `torch.aten`,
- canonicalized module verifies,
- test is deterministic and skips offline.

### Phase B: Model IR coverage for GPT-2 ops

Status: **partial**

Add or complete Model IR / target ops:

- embedding gather,
- positional add/broadcast,
- reshape/transpose/split,
- multi-head attention decomposition,
- final `Linear`,
- `argmax`,
- `embedding gather`.

Files:

- `compiler/model_ir.py`
- `compiler/frontends/manual_frontend.py`
- `compiler/mlir/mlir_builder.py`
- `mlir/mgpu/include/mgpu/MGPUOps.td`
- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`

Exit criteria:

- every op in the GPT-2 forward graph has either an mgpu lowering or a
  software operator fallback,
- unsupported ops fail with op/shape/dtype diagnostics.
- every software kernel has a differential test against its mgpu
  implementation where the mgpu shape is supported.

Remaining:

- wire pretrained weights into the full Model IR graph and validate the
  resulting software execution against PyTorch.

### Phase C: Multi-head attention

Status: **done**

Implement a decomposed attention path:

```text
q,k,v projections
  -> split into (batch, heads, seq, head_dim)
  -> q @ k^T
  -> scale
  -> causal mask + masked softmax
  -> attn @ v
  -> merge heads
  -> output projection
```

First use scalar/grid-aware matmul.  Add `mgpu.mma` as an optimization later.
Keep the software attention implementation as the fast full-model path and
the mgpu decomposed attention as the correctness oracle for small shapes.

Files:

- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`
- `compiler/mlir/mlir_builder.py`
- `tests/operator_e2e/test_attention_kernels_e2e.py`

Exit criteria:

- `batch=1`, `heads=12`, `seq<=16`, `head_dim=64` attention matches PyTorch,
- software attention matches PyTorch on larger GPT-2-like shapes,
- mgpu attention matches software attention on a cropped equivalent shape,
- causal mask produces correct triangular attention,
- ISA uses bounded loops rather than unrolled head dimensions.

Remaining:

- none; this phase is complete.

### Phase D: Large linear/matmul execution

Status: **software partial**

Generalize the existing `Linear` and scalar matmul paths:

- grid over all output elements,
- loop over `k`,
- wave execution over SMs,
- optional tiled MMA when shapes align.

Files:

- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`
- `tests/operator_e2e/test_hardware_scaling_e2e.py`

Exit criteria:

- `768x768` and `768x3072` linear layers run correctly,
- generated ISA size is bounded,
- runtime is acceptable for `seq=8`.

Remaining:

- full mgpu numerical execution of GPT-2-sized linear layers is gated by
  Python simulator speed; software execution and bounded lowering tests are
  in place.

### Phase E: Model-wide memory planning

Status: **done**

Use the lifetime-aware planner for:

- embeddings,
- block weights,
- activations,
- final logits,
- KV cache in the cache-decode milestone.

Files:

- `runtime/memory_plan.py`
- `compiler/mlir/mgpu_runner.py`
- `runtime/driver.py`
- `tests/model_e2e/test_real_llm_memory.py`

Exit criteria:

- peak memory fits configured global memory,
- memory plan is deterministic and reproducible,
- planner reports total and peak bytes.

The test builds an explicit GPT-2-sized buffer plan for embeddings, all 12
transformer-block weights, activations, final logits, and the KV cache. Full
module-level plan injection now passes bufferization after `tm_tensor.scan`
and the initial `linalg.generic` pattern lowering work. It is still not the
primary full-model execution path: remaining fused reductions such as
softmax/argmax are lower priority than completing the software operator
runner.

### Phase F: Autoregressive loop

Status: **not started**

Implement:

- prompt encoding,
- repeated one-token decode steps,
- `argmax` token selection,
- KV cache append/gather after Phase C,
- run several steps through the software operator runner and collect token
  IDs.

Files:

- `runtime/driver.py`
- `examples/transformer/autoregressive_loop.py`
- `tests/model_e2e/test_real_llm_decode.py`

Exit criteria:

- generated token IDs match PyTorch greedy decode through the software
  operator runner,
- each step uses the runtime launch queue,
- KV cache grows without corrupting previous tokens.

### Phase G: Numerical and performance gates

Status: **not started**

Implement:

- comparison against PyTorch for logits, attention, and token IDs on the
  software operator path,
- cycle/profile checks for mgpu operator gates where supported,
- a cropped smoke test and an opt-in full test,
- tolerances documented in `docs/tolerances.md`.

Files:

- `tests/model_e2e/`
- `docs/tolerances.md`

Exit criteria:

- full static `seq=8` software operator forward matches PyTorch,
- greedy decode matches PyTorch token IDs,
- pre-commit, unittest, and lit are green.

## 5. Risks

- **Multi-head attention complexity.**  This is the largest remaining
  lowering task. The software attention path is the full-model workhorse;
  mgpu attention remains a cropped correctness oracle.
- **Memory size.**  FP32 GPT-2 weights are large.  Use typed storage or
  FP16/BF16 after correctness.
- **Python simulator speed.**  Keep the first gate at `seq=8` and
  prefer the native software operator runner for full-model execution;
  optimize mgpu execution only after correctness.
- **Function-boundary bufferization.**  Full canonicalized modules may need
  explicit `one-shot-bufferize` options before map-to-mgpu. Explicit software
  buffer planning avoids this dependency for the first full-model gate.

## 6. Deliverable

`tests/model_e2e/test_real_llm_decode.py` runs a real pretrained GPT-2-small
forward and greedy decode loop through the software operator runner and
compares generated token IDs with PyTorch. mgpu operator tests remain the
correctness oracle for individual supported kernels.
