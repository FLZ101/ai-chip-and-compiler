# Session Handoff

## Repository

`/data/work/ai-chip-and-compiler`

Working tree is clean. The most recent commit is:

```text
6e93fc6 Use project-local ResNet18 cache
```

The surrounding ResNet18 phase commits are:

```text
2a1f581 Implement ResNet18 phase E gates and tolerances
a07fe60 Implement ResNet18 phase D end-to-end path
5b8d880 Implement ResNet18 phase C model memory planning
55e4a69 Implement ResNet18 phase B scalable conv and pooling
01962fd Implement ResNet18 phase A full-module pipeline
4d881ce Add ResNet18 full execution plan
```

## What is currently done

`docs/resnet18-run-plan.md` phases A through E are implemented.

- Phase A: full canonicalized ResNet18 lowers through
  `one-shot-bufferize`, `map-to-mgpu`, and `mgpu-to-isa`; `run_model_pipeline`
  returns a multi-program `MGpuIsaBundle`.
- Phase B: scalar conv loop handles stride/pad/dilation; pooling and global
  average pooling are loop-based; 224x224 conv compiles to bounded ISA.
- Phase C: `run_model_pipeline_with_memory_plan` builds a deterministic
  lifetime-aware `MemoryPlan`, injects `mgpu.memory_layout`, and the driver can
  materialize planned buffers with `Driver.allocate_from_plan`.
- Phase D: full ResNet18 FX graph conversion, full lowering, final classifier
  ISA gate, and an opt-in full forward test.
- Phase E: cropped pretrained stem profile gate and `docs/tolerances.md`.

The full 224x224 numerical execution is implemented but intentionally skipped
by default because the Python functional simulator is too slow. To run it:

```bash
MGPU_FULL_RESNET18=1 python -m unittest \
  tests.model_e2e.test_real_cnn_execution.TestRealCNNExecution.test_full_resnet18_forward_runs_in_simulator
```

## Verification commands

Run these from `/data/work/ai-chip-and-compiler`.

```bash
source .venv/bin/activate

# C++ backend, if needed
cmake --build .deps/mgpu-build --target mgpu-opt -j8

# Python suite
python -m unittest discover -v

# MLIR lit tests
lit -q tests/mlir

# Formatting/pre-commit
pre-commit run --all-files
```

Latest known results:

- Python: 192 tests, 1 skipped, all pass
- MLIR lit: 37 tests, all pass
- pre-commit: pass

## Local model caches

ResNet18 weights are stored in the project-local, gitignored cache:

```text
.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
```

The old home cache `~/.cache/torch` was removed. Tests now prefer the
project-local path and fall back to the home path if necessary.

GPT-2-small is also cached at:

```text
.cache/gpt2-small/config.json
.cache/gpt2-small/pytorch_model.bin
```

## Key files

- `docs/resnet18-run-plan.md`
- `docs/gpt2-small-run-plan.md`
- `docs/next-stage-steps.md`
- `compiler/mlir/mgpu_runner.py`
- `compiler/frontends/torch_fx_frontend.py`
- `compiler/mlir/mlir_builder.py`
- `mlir/mgpu/include/mgpu/MGPUOps.td`
- `mlir/mgpu/lib/mgpu/MapToMGPUPass.cpp`
- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`
- `runtime/memory_plan.py`
- `runtime/driver.py`
- `tests/model_e2e/test_real_cnn_execution.py`
- `tests/model_e2e/test_real_pretrained_cnn.py`

## Likely next work

1. Run and debug the opt-in full ResNet18 224x224 forward if full numerical
   validation is needed.
2. Continue the GPT-2-small plan in `docs/gpt2-small-run-plan.md`, especially
   multi-head attention and KV-cache execution.
3. Continue hardware-scaling work documented in the hardware-scaling and
   next-stage plans.
