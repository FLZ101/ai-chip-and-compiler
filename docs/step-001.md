# Step 001: First Simulator Vertical Slice

This document describes the first implemented vertical slice in the repository.
Step 002 (branches, divergence, and reconvergence) and Step 003 (block barriers
and multi-warp scheduling), and Step 004 (timing model and performance counters)
and Step 005 (unified Model IR and manual frontend) and Step 006 (MLIR builder
and `mgpu` dialect) and Step 007 (shared operator E2E harness), Step 008
(common kernels), Step 009 (activation and shape/layout kernels), Step 010
(CNN kernels), Step 011 (CNN normalization and detection helpers), Step 012
(Transformer normalization and softmax), Step 013 (attention, RoPE, and
KV-cache), Step 014 (ONNX and PyTorch FX frontends), and Step 015 (runtime and
driver), and Step 016 (CNN/object-detection model examples) are now also
implemented.
They are described in [step-002.md](step-002.md), [step-003.md](step-003.md),
[step-004.md](step-004.md), [step-005.md](step-005.md), and
[step-006.md](step-006.md), [step-007.md](step-007.md), and
[step-008.md](step-008.md), [step-009.md](step-009.md),
[step-010.md](step-010.md), [step-011.md](step-011.md), and
[step-012.md](step-012.md), [step-013.md](step-013.md), and
[step-014.md](step-014.md), [step-015.md](step-015.md), and
[step-016.md](step-016.md).

The first milestone is intentionally narrow: prove that we can parse a small GPU ISA, model global/shared memory, execute a simple SIMT warp, and run an end-to-end vector-add program against a deterministic simulator.

---

## 1. What is implemented

### 1.1 ISA

File: [../hw/isa.py](../hw/isa.py)

- Parses compact assembly text into structured `Instruction` objects.
- Supports three operand kinds:
  - `Reg`: normal and special registers.
  - `Imm`: integer or floating-point immediate.
  - `Mem`: base-plus-byte-offset memory operand.
- Supports ALU, memory, control, and special-function opcodes.
- Performs early validation for register and memory destinations.
- Formats parsed programs back to assembly.
- Reserves labels for future branch support.

Supported opcode groups:

| Group | Opcodes |
| --- | --- |
| ALU | `MOV`, `ADD`, `SUB`, `MUL`, `DIV`, `NEG`, `ABS`, `MAX`, `MIN`, `FMA` |
| Special math | `RCP`, `RSQRT`, `EXP2`, `LOG2` |
| Memory | `LDG`, `STG`, `LDS`, `STS` |
| Control | `EXIT`, `RET`, `BAR.SYNC`, `BRA` reserved |

Special registers:

- `tid.x`, `tid.y`, `tid.z`
- `ntid.x`, `ntid.y`, `ntid.z`
- `ctaid.x`, `ctaid.y`, `ctaid.z`
- `nctaid.x`, `nctaid.y`, `nctaid.z`
- `laneid`, `warpid`
- `clock`

### 1.2 Memory model

File: [../hw/memory.py](../hw/memory.py)

- `ByteMemory` models byte-addressable global or shared memory.
- FP32 access is implemented as little-endian 4-byte reads and writes.
- All accesses have bounds checks.
- `RegisterFile` provides per-thread registers `R0..R63`.
- `as_int` rejects non-integral address values.

### 1.3 SIMT execution core

File: [../hw/core.py](../hw/core.py)

Implemented hardware abstractions:

- `LaneState`: one thread lane with its own registers.
- `WarpState`: one warp with lanes, program counter, active mask, and exit state.
- `SM`: one streaming multiprocessor with shared memory and resident warps.
- `Device`: top-level GPU with global memory and multiple SMs.

Execution loop:

```text
device.clock += 1
for each SM:
    for each warp:
        if not exited:
            execute current instruction for all active lanes
```

Implemented instruction execution:

- global and shared loads/stores,
- scalar ALU operations per active lane,
- special register evaluation,
- `EXIT` and `RET`,
- `BAR.SYNC` accepted as a no-op placeholder.

### 1.4 Public device API

File: [../hw/device.py](../hw/device.py)

Exports:

- `create_device`
- `launch`
- `Device`

This keeps the execution-core implementation behind a stable public interface.

---

## 2. End-to-end test

File: [../tests/operator_e2e/test_vector_add_e2e.py](../tests/operator_e2e/test_vector_add_e2e.py)

The first full test executes this ISA program:

```asm
MUL R0, tid.x, 4
LDG R1, [R0]
LDG R2, [R0 + 4096]
ADD R3, R1, R2
STG [R0 + 8192], R3
EXIT
```

Flow:

1. Create a device.
2. Write input vectors to global memory.
3. Parse the assembly.
4. Launch one block with `n` threads.
5. Execute the kernel.
6. Read output floats from global memory.
7. Compare with the expected result.

The test also verifies that identical launches produce identical cycle counts.

---

## 3. ISA unit tests

File: [../tests/unit/test_isa.py](../tests/unit/test_isa.py)

Tests cover:

- vector-add program parsing,
- special-function instruction parsing,
- assembly parse/format round-trip,
- rejection of non-register destinations,
- rejection of duplicate labels.

---

## 4. Project tooling

### Packaging

File: [../pyproject.toml](../pyproject.toml)

- PEP 621 project metadata.
- Runtime dependencies: `numpy`, `onnx`.
- Optional groups: `test`, `frameworks`, `hardware`, `dev`.
- Pytest configuration.
- Black configuration.

### Pre-commit

File: [../.pre-commit-config.yaml](../.pre-commit-config.yaml)

- Runs Black through the official Black pre-commit mirror.
- Installed in the local Git repository.

### Virtual environment

The project uses a local `.venv`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

---

## 5. What is intentionally not implemented yet

The current simulator is the first vertical slice plus branch, barrier,
multi-warp, and timing-model support. It does not yet include:

- multiple blocks per SM over time,
- L1/L2 cache modeling,
- tensor-core `MMA` execution,
- FP16/BF16/FP8/INT8/INT4 support,
- a Model IR or MLIR compiler pipeline.

These items are described in [gpu.md](gpu.md), [types.md](types.md), and the overall [PLAN.md](PLAN.md).

---

## 6. How to run

```bash
source .venv/bin/activate
python -m unittest discover -v
```

Expected result: all current tests pass.

---

## 7. Next step

The next step is to add Transformer/LLM model examples.  This is described as
Step 017 in [steps.md](steps.md).
