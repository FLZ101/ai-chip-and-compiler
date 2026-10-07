# Step 004: Timing Model and Performance Counters

This document describes the fourth simulator milestone.

Step 004 replaces the original ``one instruction per warp per cycle`` loop with
a small, deterministic stall-aware timing model.  Functional execution is
unchanged; the new model adds latency, bandwidth, scoreboarding, and profile
counters on top of the existing ISA.

---

## 1. Timing model

The model is implemented in [../hw/timing.py](../hw/timing.py).

Every instruction belongs to one timing class:

| Class | Opcodes | Default latency |
| --- | --- | --- |
| ALU | `MOV`, `ADD`, `SUB`, `MUL`, `NEG`, `ABS`, `MAX`, `MIN`, `FMA`, `CMP*`, `SEL` | 4 cycles |
| Special | `DIV`, `RCP`, `RSQRT`, `EXP2`, `LOG2` | 8 cycles |
| Global memory | `LDG`, `STG` | 20 cycles |
| Shared memory | `LDS`, `STS` | 4 cycles |
| Control | `BRA`, `RET`, `EXIT` | 1 cycle |
| Barrier | `BAR.SYNC` | 2 cycles |

The defaults are centralized in `TimingModel`, so tests or future kernels can
construct a device with custom values:

```python
from hw.device import create_device
from hw.timing import TimingModel

model = TimingModel(global_memory_latency=40)
device = create_device(timing_model=model)
```

Global memory bandwidth defaults to 128 bytes/cycle, which is exactly one full
32-lane FP32 warp access (32 lanes x 4 bytes).

---

## 2. Per-SM issue scheduling

Each SM has one instruction issue slot per cycle.

At the start of each device cycle, the scheduler:

1. Resolves any pending reconvergence transitions for non-waiting warps.
2. Selects one ready warp using round-robin order.
3. Issues that warp's current instruction.
4. Counts one stall for every other non-exited warp.

This makes multi-warp behavior more realistic than the step 003 loop, where
every warp advanced once per global cycle.

---

## 3. Register scoreboard

Each lane tracks the cycle at which each normal register write becomes visible.

- A register write is made functionally at issue time for simplicity.
- The scoreboard prevents a dependent instruction from reading the register
  before its configured latency has passed.
- Special registers (`tid.x`, `ctaid.x`, `laneid`, and so on) do not use the
  scoreboard because they are provided by the scheduler.

A read-after-write chain is therefore reported as `scoreboard` stalls rather
than being folded into generic ALU stalls.

---

## 4. Memory traffic and shared-memory banks

The profile records logical memory traffic separately:

- `global_load_bytes`
- `global_store_bytes`
- `shared_load_bytes`
- `shared_store_bytes`

Shared memory is modeled as 32 banks of 4 bytes.  For `LDS` and `STS`, the
simulator maps each active lane's address to a bank and reports
`shared_bank_conflicts` as the number of extra bank accesses.  Bank conflicts
are reported but do not currently add extra cycle stalls.

---

## 5. Profile API

Pass `profile=True` to `launch` to receive a snapshot:

```python
from hw.device import create_device, launch
from hw.isa import parse_asm

asm = """
MOV R0, 0
LDG R1, [R0]
EXIT
"""

device = create_device()
profile = launch(device, parse_asm(asm), profile=True)
print(profile.as_dict())
```

The snapshot is a deep copy, so it remains stable if the device is launched
again.  `device.profile` is also available after execution.

`ProfileCounters` contains:

- `cycles`
- `issued_instructions`
- `stalls.fetch`
- `stalls.alu`
- `stalls.memory`
- `stalls.barrier`
- `stalls.scoreboard`
- `global_load_bytes`
- `global_store_bytes`
- `shared_load_bytes`
- `shared_store_bytes`
- `shared_bank_conflicts`

---

## 6. Files changed

- [../hw/timing.py](../hw/timing.py) (new)
- [../hw/core.py](../hw/core.py)
- [../hw/device.py](../hw/device.py)
- [../tests/operator_e2e/test_timing_model_e2e.py](../tests/operator_e2e/test_timing_model_e2e.py) (new)

---

## 7. Tests

New tests cover:

- deterministic profile snapshots across repeated launches,
- ALU-bound versus memory-bound stall profiles,
- register scoreboard stalls,
- barrier stalls with asymmetric warp arrival,
- shared-memory bank-conflict reporting.

Run:

```bash
source .venv/bin/activate
python -m unittest discover -v
```

All existing tests continue to pass.
