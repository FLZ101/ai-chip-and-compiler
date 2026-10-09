# C++ Simulator Plan

## 1. Goal

Replace the Python functional simulator hot path with a C++ execution engine
that is deterministic, exposes the same ISA and timing model, and is fast
enough to run full ResNet18 `1x3x224x224` numerical validation.

The current Python simulator is correct for operator tests but is too slow for
the full-model path. The attempted full ResNet18 run remained CPU-bound in
`hw/core.py:_alu` and had to be interrupted. A C++ simulator removes the
per-instruction Python interpreter overhead while preserving observable
semantics.

## 2. Non-goals

- Do not change the ISA text format or opcode semantics.
- Do not change the MLIR/compiler pipeline.
- Do not make the simulator multi-threaded initially; determinism is more
  important than parallelism.
- Do not replace the Python driver/runtime API immediately. The Python
  `runtime.Driver`, `Buffer`, and `MemoryPlan` APIs should keep working.

## 3. Architecture

```text
Python compiler/runtime
          |
          v
pybind11 wrapper
          |
          v
C++ Simulator
  - ISA parser / Program
  - ByteMemory / typed access
  - Lane / Warp / SM / Device
  - Block wave scheduler
  - Timing model / ProfileCounters
```

The C++ simulator lives under a new tree, for example:

```text
cpp_sim/
  CMakeLists.txt
  include/cpp_sim/
  src/
  tests/
```

The existing Python `hw/` modules become thin wrappers when
`MGPU_CPP_SIM=1` is set. If the native library is unavailable, tests fall back
to the current Python simulator.

## 4. Implementation phases

### Phase 0: C++ ISA parser

Implement a C++ parser that matches `hw/isa.py` exactly.

Deliverables:

- `Program` with instructions and resolved labels.
- `Reg`, `Imm`, `Mem`, `Label` operand types.
- Same special registers and opcode sets.
- Same parsing errors for malformed instructions.

Exit criteria:

- every `.asm` accepted by Python is accepted by C++,
- every malformed Python test is rejected with the same diagnostic class.

### Phase 1: C++ memory subsystem

Implement byte-addressable global/shared memory with typed loads/stores:

- `f32`, `f16`, `bf16`
- `f8_e4m3`, `f8_e5m2`
- `i8`, `u8`, `i4`, `u4`, `i32`

Deliverables:

- bounds-checked `ByteMemory`
- typed read/write methods matching `hw/memory.py`
- packed INT4/UINT4 semantics

Exit criteria:

- cross-check typed accesses against Python `ByteMemory`
- packed i4/u4 edge cases match bit-for-bit

### Phase 2: SIMT execution core

Implement:

- 32-lane warps,
- active masks and branch reconvergence,
- special registers (`tid`, `ctaid`, `ntid`, `nctaid`, `laneid`, `warpid`,
  `clock`),
- registers with scoreboard-ready state,
- `BRA`, `EXIT`, `BAR.SYNC`,
- all ALU, special-function, load/store, MMA, and quantization opcodes.

Deliverables:

- `Device`, `SM`, `Warp`, `Lane`
- wave block scheduler with configurable `sm_count`
- deterministic round-robin issue order

Exit criteria:

- differential tests against Python for small and medium kernels,
- divergence/barrier/3D-index tests pass identically.

### Phase 3: Timing and profiling

Reimplement `TimingModel`, `StallCounters`, and `ProfileCounters` in C++.

Exit criteria:

- profile counters match Python for all current timing tests,
- repeated launches produce identical counters.

### Phase 4: pybind11 compatibility layer

Expose:

- `create_device(global_size, shared_size, sm_count, register_count)`
- `launch(device, asm_text, grid_dim, block_dim, profile)`
- typed memory read/write helpers used by tests,
- profile counter snapshots.

The Python package should expose the same `hw.device` API.

Exit criteria:

- `MGPU_CPP_SIM=1 python -m unittest discover` passes,
- `lit -q tests/mlir` passes,
- existing operator tests are unchanged.

### Phase 5: Performance pass

Optimize only after correctness is locked:

- predecode labels and operands once,
- keep register files as fixed-size arrays, not maps,
- avoid per-lane allocation on each instruction,
- implement address computation inline,
- optionally use compiler intrinsics for FP8/FP16 conversion,
- profile and target the full ResNet18 forward.

Deliverable:

- a benchmark harness comparing Python and C++ simulator wall-clock time.

### Phase 6: Full-model validation

Run:

```bash
MGPU_FULL_RESNET18=1 MGPU_CPP_SIM=1 python -m unittest \
  tests.model_e2e.test_real_cnn_execution.TestRealCNNExecution.test_full_resnet18_forward_runs_in_simulator -v
```

Exit criteria:

- full `224x224` ResNet18 forward finishes,
- logits match PyTorch within `atol=5e-3`, `rtol=5e-3`,
- cycles/profile are captured,
- repeated runs are deterministic.

## 5. Test strategy

Reuse the existing Python test suite as a differential oracle.

For every operator test:

1. Run through Python simulator.
2. Run through C++ simulator.
3. Compare outputs, profile counters, and error behavior.

Add targeted native unit tests for:

- ISA parser,
- typed memory,
- MMA fragment layout,
- warp divergence,
- barrier convergence,
- wave scheduling.

## 6. Build and integration

The project already builds `mgpu-opt` with CMake under `.deps/mgpu-build`.
The C++ simulator can use a similar out-of-tree build:

```bash
cmake -S cpp_sim -B .deps/cpp-sim-build
cmake --build .deps/cpp-sim-build -j8
```

`pyproject.toml` or `Makefile` should gain targets for building and testing the
native simulator.

## 7. Performance target

Current blocker:

```text
full ResNet18 224x224 Python simulator: >7 minutes without completing
```

Target for C++ simulator:

```text
full ResNet18 224x224: completes in normal opt-in test time
```

The exact target should be measured in Phase 5. A reasonable first target is
`< 2 minutes` on the current development container, then reduce further if
needed.

## 8. Risks

- **Semantic drift:** Python and C++ may disagree on edge cases. Mitigate with
  differential tests and keep the Python simulator as oracle.
- **Typed FP8/FP16 conversion differences:** use the same conversion rules or
  reuse the existing Python `ml_dtypes` outputs as fixtures.
- **MMA layout complexity:** keep the existing Python MMA fragment-layout
  tests as source of truth.
- **Build complexity:** keep the native library optional so the repo still
  works without it.
