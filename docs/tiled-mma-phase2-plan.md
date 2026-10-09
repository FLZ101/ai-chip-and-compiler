# Tiled MMA Phase 2 Plan

## Goal

Implement the bounded, tiled Q @ K^T score stage of the general attention
kernel. Phase 2 does not implement softmax or value accumulation; those remain
separate later phases.

## Current status

Partial. The committed helpers cover lane-index setup, zeroing a fragment,
shared-memory load/store for a 16x16 tile, query-tile and key-tile loop
headers, tiled Q and K^T fragment loads, `MMA` accumulation, and full-tile
shared-memory stores.

The existing large-shape attention path still uses `emitScalarAttentionIsa`
and passes a numerical test for `seq=8`, `head_dim=64`. That scalar fallback
is intentionally kept until Phase 3/4 can consume tiled score tiles.

Still missing:

- reduction/head-dimension tile loop,
- a score-only tiled emitter,
- a test hook that exercises that emitter,
- FileCheck coverage for tiled loops and `MMA`.

## Shape model

```text
Q: memref<batch x heads x seq x head_dim x f32>
K: memref<batch x heads x seq x head_dim x f32>
```

Phase 2 targets `batch=1` first. The same loop structure later extends to
arbitrary batch/head by adding block mapping in Phase 5.

## Tile configuration

```text
tile_size = 16
q_tiles = ceil(seq / tile_size)
k_tiles = ceil(seq / tile_size)
d_tiles = ceil(head_dim / tile_size)
```

`head_dim` must be tiled too. This is the important correction to the
original sketch: a single `MMA` can reduce only a 16-element head-dimension
tile, so `head_dim=64` requires four reduction tiles.

## Correct algorithm

For each query tile and each key tile, accumulate a 16x16 score tile over all
head-dimension tiles:

```text
for b in batch:
  for h in heads:
    for q_tile in q_tiles:
      for k_tile in k_tiles:
        zero accumulator R16..R23

        for d_tile in d_tiles:
          load Q_tile[q_tile, d_tile]   -> R0..R7
          load K^T_tile[d_tile, k_tile] -> R8..R15
          MMA R16, R0, R8, R16

        store score[q_tile, k_tile] to shared memory
        BAR.SYNC
```

The query and key loops tile the output score matrix. The innermost loop is
the reduction over `head_dim`.

## ISA register convention

The emitter must stay below the 64-register limit.

```text
R0..R7    Q / A fragment
R8..R15   K^T / B fragment
R16..R23  score accumulator fragment
R24/R25   lane row/column, set once by emitLaneIndex
R26..R37  scratch used by fragment load/store helpers
R38       dim-tile loop predicate
R39       scratch immediate/address temporary
R40..R42  query-tile loop counter, limit, predicate
R43..R45  key-tile loop counter, limit, predicate
R46..R47  dim-tile loop counter and limit
```

The fragment helpers overwrite `R26..R37`, `R39`, and `R42`, which is safe
because those values are not live across a fragment load. Loop counters must
not be placed in those scratch registers.

## Shared-memory layout

Store one padded 16x16 score tile per `(q_tile, k_tile)`:

```text
score_tile_bytes = 16 * 16 * 4
score_offset(q, k) = (q * k_tiles + k) * score_tile_bytes
```

Padded rows/columns outside `seq` are zero. Later phases read this tile-major
layout when computing softmax and accumulating values.

## Integration and test strategy

Phase 2 is score-only, and the public `mgpu.attention` op must still produce a
complete attention output. Therefore do not replace the scalar fallback with
an incomplete tiled path.

Add a temporary pass option such as `--mgpu-emit-attention-score` to
`mgpu-to-isa`. When enabled, large rank-4 attention ops emit only the tiled
score stage into shared memory and exit. Normal lowering is unchanged.

This test hook lets FileCheck verify the exact tiled ISA without pretending
the full attention kernel is complete. Numerical score validation is deferred
to Phase 3, when the score tiles are consumed by softmax and can be compared
through the complete attention result.

## Steps

Steps 1-2 are already committed. Steps 3-6 are remaining.

### Step 1: Emit outer query-tile loop

Generate bounded query-tile loop labels and counters.

Files:

- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`

Exit:

- `seq=8` emits one query-tile iteration,
- `seq=32` emits two query-tile iterations.

### Step 2: Emit inner key-tile loop

Generate bounded key-tile loop labels and counters.

Files:

- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`

Exit:

- ISA contains `key_tile_loop` labels,
- no unrolled `seq`.

### Step 3: Emit reduction/dim-tile loop

Add `emitDimTileLoopStart` and `emitDimTileLoopEnd`, using the register
convention above.

Exit:

- `head_dim=64` emits four reduction-tile iterations,
- `head_dim=32` emits two.

### Step 4: Wire existing fragment helpers into a score emitter

Add `emitTiledAttentionScoreIsa` and generate:

```text
query_tile_loop
  key_tile_loop
    dim_tile_loop
      load Q fragment
      load K^T fragment
      MMA R16, R0, R8, R16
    store score tile to shared
    BAR.SYNC
```

Files:

- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`

Exit:

- padded lanes outside `seq`/`head_dim` load zero,
- score accumulator is updated across reduction tiles,
- score tile is stored at the offset above.

### Step 5: Add the test-only pass option

Wire `--mgpu-emit-attention-score` so tests can lower score-only attention
without changing the default full-attention path.

Exit:

- normal large attention still emits scalar fallback,
- score-only mode emits tiled loops and no softmax/value copy.

### Step 6: Add FileCheck coverage

Test:

- `seq=8`, `head_dim=64`,
- `seq=32`, `head_dim=32`.

Files:

- `tests/mlir/mgpu_to_isa_attention_tiled.mlir`

Exit:

- generated ISA contains `query_tile_loop`, `key_tile_loop`,
  `dim_tile_loop`, `MMA`, and `BAR.SYNC`,
- padded lanes are zeroed.

## Suggested commit order

```text
1. dim-tile loop helper
2. score-only tiled emitter
3. test-only pass option
4. FileCheck coverage
```

## Later phases

Phase 3 consumes the score tiles for causal masking and softmax. Phase 4 adds
tiled value accumulation. At that point the score-only test hook can be
removed and numerical score checks become part of the complete attention
differential test.
