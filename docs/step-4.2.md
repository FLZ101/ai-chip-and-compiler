# Step 4.2: Add Fragment Layout and Shared-Memory Swizzle

This document describes the Phase 4.2 implementation.

The simulator now has two tensor-core data-layout concepts that were deferred
from Phase 4.1:

1. An explicit A/B/C fragment layout for a square ``16x16x16`` MMA tile.
2. A PTX-style shared-memory swizzle used by bank-conflict accounting.

Both pieces are deterministic and live in the hardware model.  The compiler
pass in Phase 4.3 will use the fragment layout when lowering tiled matmuls to
``mgpu.mma``; the simulator already uses the swizzle when profiling shared
loads and stores.

## 1. Fragment layout

The new module [../hw/mma.py](../hw/mma.py) defines:

- `FragmentRole.A`, `.B`, and `.C`;
- `FragmentLayout`;
- `DEFAULT_MMA_FRAGMENT_LAYOUT`.

For the MVP, every fragment is a ``16x16`` matrix and a warp contains 32
lanes.  The lane/register mapping is:

```text
lane     = (row % 8) * 4 + (col % 4)
register = (row // 8) * 4 + (col // 4)
```

Therefore each lane owns eight elements for each of the A, B, and C roles.
This is intentionally simple, but it captures the important property that the
later tensor-core path must satisfy: every logical element is assigned to
exactly one ``(lane, register)`` slot.

## 2. Shared-memory swizzle

`shared_bank_index` maps a shared-memory byte address to a physical bank:

```text
word          = address // 4
swizzled_word = word ^ (((word >> 7) & 0x7) << 4)
bank          = swizzled_word & 31
```

The functional simulator still accesses shared memory by logical address.
Only the bank-conflict accounting uses the swizzled bank index.  This mirrors
the separation used by real GPU functional models: address swizzle affects
performance/counters, not the contents of memory.

`TimingModel` gained `shared_swizzle_bits`, defaulting to `3`.  Setting it to
`0` restores the previous linear bank mapping.

## 3. Tests

- [../tests/unit/test_mma_layout.py](../tests/unit/test_mma_layout.py) checks
  that all three fragment roles cover their 256 elements exactly once and that
  the lane-to-offset mapping is inverse-consistent.
- [../tests/operator_e2e/test_timing_model_e2e.py](../tests/operator_e2e/test_timing_model_e2e.py)
  now verifies that `shared_bank_conflicts` is deterministic across repeated
  launches.

## 4. Files changed

- [../hw/mma.py](../hw/mma.py) (new)
- [../hw/core.py](../hw/core.py)
- [../hw/timing.py](../hw/timing.py)
- [../tests/unit/test_mma_layout.py](../tests/unit/test_mma_layout.py) (new)
- [../tests/operator_e2e/test_timing_model_e2e.py](../tests/operator_e2e/test_timing_model_e2e.py)
