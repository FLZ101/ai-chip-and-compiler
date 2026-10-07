# Step 003: Block Barriers and Multi-Warp Scheduling

This document describes the third simulator milestone.

Step 003 changes the simulator from a single warp per block into a more
realistic multi-warp SIMT model and adds a real block-scoped barrier.

---

## 1. Multi-warp launch

When a block is launched:

1. `block_threads = prod(block_dim)`.
2. Threads are grouped into 32-lane warps.
3. The last warp may contain fewer than 32 lanes.
4. Each warp receives an active mask covering only its existing lanes.
5. All warps in a block are resident on the same SM.

For example, `block_dim = (64,)` creates two warps:

- warp 0: lanes 0..31
- warp 1: lanes 0..31

Global thread IDs are derived as:

```text
tid.x = block_linear_id * block_threads
        + warp_id * 32
        + lane_id
```

## 2. Three-dimensional launch indices

The simulator also exposes 3D forms of the special registers:

- `ctaid.x`, `ctaid.y`, `ctaid.z`
- `tid.x`, `tid.y`, `tid.z`
- `nctaid.x`, `nctaid.y`, `nctaid.z`
- `ntid.x`, `ntid.y`, `ntid.z`

Dimensions that are omitted from a launch tuple are treated as size 1, and
their indices are 0. This matches CUDA-style default 1D behavior.

---

## 3. Block barrier

`BAR.SYNC` now has real semantics:

1. The executing warp marks itself as waiting.
2. The warp stops issuing instructions.
3. When every warp in the SM block arrives, all waiting warps are released.
4. Each released warp advances its PC past the barrier.

This makes shared-memory writes from all warps visible after the barrier.

---

## 4. Files changed

- [../hw/core.py](../hw/core.py)
- [../tests/operator_e2e/test_barrier_multiwarp_e2e.py](../tests/operator_e2e/test_barrier_multiwarp_e2e.py)
- [../tests/operator_e2e/test_3d_index_e2e.py](../tests/operator_e2e/test_3d_index_e2e.py)

---

## 5. Tests

New tests cover:

- two-warp vector add,
- shared-memory writes followed by `BAR.SYNC`,
- cross-warp reads after the barrier,
- 3D grid/block index calculation,
- deterministic execution across repeated launches.

Run:

```bash
source .venv/bin/activate
python -m unittest discover -v
```

All existing tests continue to pass.
