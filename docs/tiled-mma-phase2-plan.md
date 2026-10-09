# Tiled MMA Phase 2 Plan

## Goal

Implement the tiled Q @ K^T score-computation portion of the general tiled
attention kernel.

## Inputs

```text
Q: memref<batch x heads x seq x head_dim x f32>
K: memref<batch x heads x seq x head_dim x f32>
```

For the first implementation, `batch=1`. The same loops should later extend to
arbitrary batch/head.

## Tile configuration

```text
tile_size = 16
q_tiles = ceil(seq / tile_size)
k_tiles = ceil(seq / tile_size)
```

Each MMA computes:

```text
S_tile = Q_tile @ K_tile^T
```

## Generated ISA loop structure

```text
for b in batch:
  for h in heads:
    for q_tile in q_tiles:
      zero accumulator R16..R23

      for k_tile in k_tiles:
        load Q_tile -> R0..R7
        load K_tile^T -> R8..R15
        MMA R16, R0, R8, R16

      store score tile to shared memory
      BAR.SYNC
```

## Steps

### Step 1: Emit outer query-tile loop

Generate ISA labels and loop counters for query tiles.

Files:

- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`

Exit:

- `seq=8` emits one query tile,
- `seq=32` emits two query-tile iterations.

### Step 2: Emit inner key-tile loop

Generate inner loop over key tiles.

Files:

- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`

Exit:

- ISA contains bounded key-tile loops,
- no unrolled `seq`.

### Step 3: Load Q and K^T tiles

Use fragment staging helpers to load each tile.

Files:

- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`

Exit:

- lanes outside `seq`/`head_dim` load zero,
- Q and K^T fragments occupy expected register blocks.

### Step 4: Emit per-tile MMA

Emit:

```text
MMA R16, R0, R8, R16
```

for each key tile.

Exit:

- score accumulator is updated across key tiles.

### Step 5: Store score tile to shared memory

Store the completed score tile using the generic staging helper.

Exit:

- score tile is in shared memory,
- `BAR.SYNC` separates score production from later softmax.

### Step 6: Add lit and operator tests

Test:

- `seq=8`, `head_dim=64`,
- `seq=32`, `head_dim=32`,
- padded lanes zero.

Files:

- `tests/mlir/mgpu_to_isa_attention.mlir`
- `tests/operator_e2e/test_attention_kernels_e2e.py`

Exit:

- generated ISA contains loops and `MMA`,
- score tiles match NumPy Q@K^T.

## Suggested commit order

```text
1. outer query loop
2. inner key loop
3. Q/K tile loads
4. per-tile MMA
5. score staging
6. tests
```
