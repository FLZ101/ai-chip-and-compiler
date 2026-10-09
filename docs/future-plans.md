# Future Plans

This document is the forward-looking plan after the Python simulator MVP,
the C++ `mgpu-opt` backend, tiled MMA attention, and the GPT-2 software
forward gate.

It is deliberately organized as independent workstreams.  The first two are
the shortest path to a useful real-model demo; the remaining tracks harden the
compiler and simulator for larger workloads.

## 1. Current baseline

Already working:

- deterministic Python and C++ simulators,
- out-of-tree `mgpu-opt` C++ backend,
- `linalg`/`memref` lowering, typed memory, and tensor-core MMA,
- tiled MMA attention for arbitrary batch/head/query tiles,
- software operator runner for all GPT-2 graph operators,
- full GPT-2-small static forward gate matching PyTorch,
- GPT-2 memory plan and Phase 7 runtime hardening,
- ResNet18 import/memory-planning and software forward gates.

Still missing before the stack feels like a complete MVP:

- greedy GPT-2 decode loop with KV cache,
- full GPT-2 through `mgpu-opt`/ISA,
- fast enough C++ execution for large models,
- complete pretrained CNN simulator execution,
- production autotuning and profiling integration.

## 2. Workstream A: GPT-2 decode and numerical gates

Goal: go from one static forward to several greedy decode steps.

Steps:

1. Build a decoder-step graph with:

   ```text
   input_ids -> embeddings
             -> transformer blocks
             -> lm_head
             -> argmax
   ```

2. Add KV-cache append/gather nodes for `use_cache=True`.
3. Run prompt encoding once, then decode token-by-token.
4. Compare generated token IDs with PyTorch greedy decoding.
5. Add logits, attention, and token-level tolerance gates.

Files:

- `tests/model_e2e/test_real_llm_decode.py`
- `runtime/driver.py`
- `examples/transformer/`

Exit criteria:

- at least 4 decode steps match PyTorch greedy IDs,
- KV cache grows without corrupting previous tokens,
- repeated runs are deterministic.

## 3. Workstream B: Full GPT-2 mgpu/ISA path

Goal: lower the complete GPT-2 graph through `map-to-mgpu` and
`mgpu-to-isa`, not just the software runner.

Steps:

1. Finish `linalg.generic` lowering for fused reductions:

   - softmax,
   - argmax,
   - `arith.index_cast`,
   - rank-3 batched linear.

2. Add `embedding_gather` as a first-class `mgpu` op or robust `linalg.generic`
   pattern.
3. Generalize `emitLinearIsa` to rank-3 inputs.
4. Run the full canonicalized GPT-2 module through the memory-plan pipeline.
5. Use the C++ simulator for selected small GPT-2 configurations.

Files:

- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`
- `mlir/mgpu/include/mgpu/MGPUOps.td`
- `compiler/mlir/mgpu_runner.py`

Exit criteria:

- full GPT-2 `seq=8` lowers to a sequence of `mgpu.isa` programs,
- a cropped mgpu execution matches the software operator runner,
- no `tm_tensor`, `arith.index_cast`, or unsupported `linalg.generic`
  diagnostics remain.

## 4. Workstream C: C++ simulator speed

Goal: make full-model execution practical enough for iterative development.

Steps:

1. Benchmark full-model software and ISA paths.
2. Avoid copying the whole global buffer per software operator.
3. Add native C++ kernels for:

   - batched linear,
   - layer norm,
   - GELU,
   - masked attention,
   - transpose/reshape.

4. Add progress/profiling hooks so large runs report per-op timing.
5. Cache compiled kernels or reuse preallocated workspaces.

Files:

- `cpp_sim/`
- `tests/operator_e2e/helpers.py`
- `scripts/benchmark_cpp_sim.py`

Exit criteria:

- full GPT-2 software forward is at least 2-3x faster than the current
  Python-only path,
- per-op profile output is stable,
- software and mgpu results remain identical.

## 5. Workstream D: Pretrained CNN execution

Goal: finish ResNet18 and add a second CNN target.

Steps:

1. Remove the opt-in skip from full ResNet18 simulator execution.
2. Run ResNet18 224x224 through the C++ simulator and compare logits.
3. Add MobileNetV3-Small import/software execution.
4. Profile convolution and memory-plan peaks.

Files:

- `tests/model_e2e/test_real_cnn_execution.py`
- `tests/model_e2e/test_real_pretrained_cnn.py`
- `docs/resnet18-run-plan.md`

Exit criteria:

- full ResNet18 logits match PyTorch within documented tolerance,
- MobileNetV3 imports and runs through the software runner,
- per-layer timings are collected.

## 6. Workstream E: Quantization and MMA performance

Goal: make quantized model execution useful, not just correct.

Steps:

1. Extend tiled MMA to INT8/FP8 matmul paths.
2. Add fused quantize/dequantize around real model weights.
3. Lower QDQ-wrapped ONNX models end to end.
4. Report MMA utilization and shared-memory conflicts.

Files:

- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`
- `hw/timing.py`
- `cpp_sim/src/core.cpp`
- `tests/operator_e2e/test_quantization_e2e.py`

Exit criteria:

- INT8/FP8 real model blocks run in the C++ simulator,
- quantized outputs match PTQ references,
- profile counters are deterministic.

## 7. Workstream F: Runtime, tooling, and reproducibility

Goal: make the project easier to operate and reproduce.

Steps:

1. Connect the runtime autotuner to `mgpu-opt` pass options.
2. Add a command-buffer API over `Driver`.
3. Add model-level progress bars and per-op logs.
4. Export `session.md` after long-running work.
5. Add a release gate:

   ```bash
   make test
   make lit
   make precommit
   make full-resnet18-software
   make gpt2-software
   ```

Files:

- `runtime/`
- `compiler/mlir/mgpu_runner.py`
- `Makefile`
- `docs/`

Exit criteria:

- `make test` and `make gpt2-software` pass from a clean checkout,
- long-running model targets print stable progress,
- session export captures the current state.

## 8. Suggested milestones

```text
M1: GPT-2 greedy decode matches PyTorch for >= 4 tokens
M2: Full GPT-2 lowers through mgpu-opt at seq=8
M3: C++ simulator full GPT-2 forward runs in practical time
M4: Full ResNet18 simulator execution is no longer opt-in
M5: Quantized real-model block matches PTQ reference
M6: clean-checkout release gate passes
```

## 9. Risks

- Full GPT-2 mgpu lowering may expose many small `linalg.generic` patterns.
  Keep the software runner as the correctness fallback.
- The current C++ software runner copies the entire global buffer per native
  op.  This must be fixed before large-model speed work is meaningful.
- Exact PyTorch matching for GPT-2 requires gelu_new semantics, correct
  Conv1D weight orientation, and correct head split/merge.  Keep those
  invariants covered by tests.
- Full CNN execution may require more global memory than the default
  simulator configuration.
