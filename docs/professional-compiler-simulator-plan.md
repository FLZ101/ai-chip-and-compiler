# Professional Compiler and GPU Simulator Plan

This document describes the long-term path from the current research/MVP
codebase to a production-grade ML compiler and cycle-approximate GPU
simulator.

The near-term product workstreams remain in
[future-plans.md](future-plans.md). This document is the professionalization
overlay: the infrastructure, invariants, and validation needed before the
project is trustworthy enough for external users.

## 1. Vision

Build a deterministic, observable, and testable stack:

```text
PyTorch / ONNX / MLIR
          |
          v
professional ML compiler
          |
          v
documented device ISA
          |
          v
cycle-approximate GPU simulator
          |
          v
golden outputs, traces, profiles, and diagnostics
```

The compiler and simulator must be independently useful:

- the compiler can be tested and benchmarked without executing hardware,
- the simulator can run hand-written or fuzzed ISA programs without the
  compiler,
- the combined stack has a deterministic reference execution path.

## 2. Engineering principles

1. **Canonical IR and verifier.**
   Every lowering produces valid IR; invalid IR is a compiler bug, not user
   input.

2. **Determinism.**
   Pass ordering, allocation, scheduling, wave assignment, and error messages
   must be reproducible.

3. **Explicit types.**
   Storage type, compute type, layout, alignment, and memory space are
   explicit.

4. **No silent fallback.**
   Unsupported operations, dynamic dimensions, dtypes, or device
   configurations fail with structured diagnostics.

5. **Layered testing.**
   Every abstraction has:

   - unit tests,
   - IR FileCheck tests,
   - differential tests against a golden model,
   - randomized/fuzz tests,
   - performance regression tests.

6. **Observability.**
   Every pass, kernel, warp, and memory transaction can be traced.

## 3. Compiler roadmap

### C1: IR and pass infrastructure

Deliverables:

- one canonical MLIR entry point,
- pass registration and pass-manager pipelines,
- verifier runs after every pass in debug mode,
- `--mlir-print-ir-after-all`, `--mlir-print-ir-before-all`,
- pass reproducers and crash reports,
- stable source locations through all lowerings,
- diagnostic severity and structured error records.

Exit criteria:

- any failing pass produces a minimal reproducer,
- debug dumps are deterministic,
- source locations survive import through ISA emission.

### C2: Frontend coverage and provenance

Deliverables:

- PyTorch FX and ONNX importers with parameter/buffer provenance,
- named tensors and stable SSA names,
- dynamic-shape metadata separated from static-shape fast paths,
- custom-op registry for vendor/target extensions,
- import-level verifier and round-trip tests.

Exit criteria:

- real CNN and LLM models import without anonymous tensors,
- imported modules parse, verify, and canonicalize deterministically.

### C3: Mid-end optimization

Deliverables:

- shape inference,
- canonicalization, CSE, DCE,
- fusion,
- tiling,
- vectorization,
- software pipelining for shared-memory copies,
- memory planning and liveness analysis.

Exit criteria:

- optimization passes preserve golden results,
- liveness and memory plans are stable,
- at least one matmul and one reduction demonstrate measurable speedup.

### C4: Backend lowering

Deliverables:

- one-shot and incremental bufferization,
- address lowering with alignment and strided memrefs,
- instruction selection and legalization,
- register allocation with spilling,
- instruction scheduling,
- ISA encoding/printing and optional assembler/disassembler,
- relocation/linking metadata for multi-kernel modules.

Exit criteria:

- backend output round-trips through assembler/parser,
- register allocation is deterministic,
- spills are exercised by tests,
- emitted code runs unchanged in both Python and C++ simulators.

### C5: Numerics and quantization

Deliverables:

- FP32/FP16/BF16/FP8/INT8/INT4 type system,
- explicit rounding, saturation, and zero-point semantics,
- fast-math flags with documented legality,
- PTQ/QAT lowering paths,
- tensor-core MMA legalization for supported dtypes,
- golden tolerance tables per dtype and op.

Exit criteria:

- quantized model blocks match PTQ references within documented tolerances,
- rounding/saturation edge cases are unit-tested,
- fast-math changes are opt-in and recorded in IR.

### C6: Compiler validation

Deliverables:

- lit/FileCheck suite,
- MLIR round-trip tests,
- differential testing versus NumPy/PyTorch,
- random program generation,
- model-zoo smoke tests,
- benchmark suite with historical JSON tracking,
- coverage and sanitizer jobs.

Exit criteria:

- `make test`, `make lit`, and `make precommit` are clean,
- fuzz jobs run in CI,
- performance regressions are detected automatically.

## 4. Simulator roadmap

### S1: ISA specification

Deliverables:

- normative ISA document,
- instruction encodings,
- assembler and disassembler,
- operand/type validation,
- reserved opcode and extension policy,
- instruction semantics reference in Python and C++.

Exit criteria:

- hand-written assembly and compiled output use the same parser,
- invalid instructions fail with source locations,
- Python and C++ simulators agree on every valid program.

### S2: Execution model

Deliverables:

- warp scheduler,
- SIMT divergence and reconvergence,
- barriers and named barriers,
- shared memory,
- local memory,
- global memory with typed access,
- async copy,
- tensor-core MMA,
- memory consistency model documentation.

Exit criteria:

- divergence, barriers, and shared-memory patterns are deterministic,
- consistency-model examples match the documented model,
- randomized warp programs match a scalar reference.

### S3: Memory hierarchy

Deliverables:

- L1/shared memory model,
- L2 cache model,
- coalescing analysis,
- bank-conflict analysis,
- DRAM bandwidth/latency model,
- alignment and swizzle handling,
- memory profiler.

Exit criteria:

- bank conflicts and coalescing counters are stable,
- cache behavior is documented,
- memory-bound kernels show expected performance trends.

### S4: Timing and power models

Deliverables:

- pipeline/scoreboard timing,
- instruction latencies and throughput,
- memory latency/bandwidth,
- occupancy and stall counters,
- power/energy estimate hooks,
- cycle-approximate and functional modes.

Exit criteria:

- repeated runs have identical cycle counts,
- ALU-bound and memory-bound kernels produce different profiles,
- performance counters can be exported as JSON.

### S5: Runtime and device model

Deliverables:

- command queue and events,
- memory allocator and lifetime planner,
- multi-SM and resident-block scheduling,
- unified addressing metadata,
- profiler, tracer, and debugger hooks,
- checkpoint/replay for long-running simulations.

Exit criteria:

- multi-kernel models run in FIFO/dependency order,
- traces can be replayed deterministically,
- long-running simulations can be resumed.

### S6: Simulator validation

Deliverables:

- Python/C++ co-simulation,
- randomized differential testing,
- memory-model invariant checks,
- performance-counter invariants,
- microbenchmark suite,
- golden trace comparison.

Exit criteria:

- Python and C++ simulators agree on fuzzed programs,
- every ISA instruction has a semantic test,
- performance counters match hand-computed microbenchmarks.

## 5. Cross-cutting engineering

Deliverables:

- C++ and Python bindings with stable APIs,
- CMake and `pyproject.toml` packaging,
- CI matrix:

  - lint,
  - formatting,
  - unit tests,
  - lit,
  - sanitizers,
  - coverage,
  - model smoke tests,
- release gates and versioning,
- changelog and API deprecation policy,
- benchmark history,
- reproducible environment lockfiles,
- documentation for architecture, ISA, passes, and simulator.

Exit criteria:

- clean checkout builds and passes release gates,
- generated artifacts are reproducible,
- external contributors can run the stack from `README` instructions.

## 6. Professional milestones

```text
P0: deterministic release gate and clean CI
P1: stable ISA spec with assembler/disassembler
P2: full verifier/debug/reproducer story
P3: cycle-approximate C++ simulator with caches and profiler
P4: register allocation, scheduling, and spills
P5: quantized and tensor-core real-model block
P6: full GPT-2 greedy decode and full ResNet18 execution
P7: differential fuzzing and model-zoo regression suite
```

## 7. Bridge from the current repo

Near-term work already planned in `future-plans.md` is still the priority:

- finish GPT-2 decode,
- finish full GPT-2 mgpu lowering,
- fix large-buffer copies in the C++ software runner,
- make full ResNet18 simulator execution default.

The professional roadmap layers infrastructure around those product goals
rather than replacing them.
