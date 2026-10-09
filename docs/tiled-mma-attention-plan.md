# Tiled MMA Attention Plan

## 1. Goal

Replace the current small-shape `mgpu.attention` MMA path with a **general**
tiled, loop-based implementation. The implementation must not hard-code
GPT-2 sizes. It should support arbitrary static shapes within simulator
resources, including:

```text
batch=1
heads=12
seq=8
head_dim=64
```

and later larger sequences.

General requirements:

- arbitrary `batch`, `heads`, `seq`, and `head_dim`,
- `head_dim` need not be a multiple of 16 for the outer algorithm,
  although MMA tiles naturally require padding internally,
- rank-2 and rank-4 Q/K/V layouts where shape information is available,
- masked and unmasked attention,
- user-supplied scale.

The generated ISA must remain bounded and must match PyTorch on GPT-2-like
shapes.

## 2. Current status

Phases 1 through 4 are implemented and tested:

- Phase 1: reusable fragment staging helpers.
- Phase 2: tiled Q @ K^T score loops over query, key, and head-dimension
  tiles.
- Phase 3: two-pass tiled softmax with explicit causal masking.
- Phase 4: tiled value accumulation and global output writes.

The original small-shape path still handles `seq <= 16` and `head_dim <= 16`.
Large rank-4 attention still falls back to `emitScalarAttentionIsa` by
default. The full tiled path is available under
`--mgpu-emit-attention-tiled` and becomes the default in a later phase.

## 3. Design

### 3.1 Tile sizes

- Query tile: `16 x 16`
- Key/value tile: `16 x 16`
- Output tile: `16 x 16`

The number of key/value tiles is computed from the shape:

```text
num_k_tiles = ceil(seq / 16)
num_v_tiles = ceil(head_dim / 16)
```

### 3.2 Algorithm

```text
for each batch/head:
  for each query tile:
    initialize running max = -inf
    initialize running sum = 0
    initialize output accumulator = 0

    for each key tile:
      score = Q_tile @ K_tile^T
      score *= scale

      if causal:
        mask keys > current query

      compute tile max
      update running max
      compute exp score
      update running sum
      score_weights = exp score / running sum

      for each value tile:
        output += score_weights @ V_tile

    write output tile
```

The first implementation may use two-pass softmax if that is simpler:

```text
pass 1: compute all score tiles and max
pass 2: compute exp and sum
pass 3: weighted value accumulation
```

Prefer the simpler two-pass version first. Optimize to online softmax later.

### 3.3 Shared-memory staging

Shared memory layout:

```text
[ score tiles ]
[ Q/K/V fragments as needed ]
```

`BAR.SYNC` is required between:

```text
score producer -> softmax consumer
softmax producer -> output producer
```

The same generated loop structure must work for any static shape. Padding
tiles that extend beyond `seq` or `head_dim` are masked or zeroed.

## 4. Implementation phases

### Phase 1: Fragment staging helpers — complete

Add helpers:

- load a `16x16` tile into MMA fragment registers,
- store a fragment to shared memory,
- load a fragment from shared memory,
- zero an accumulator tile.

Files:

- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`

Exit:

- existing small attention tests still pass,
- helper functions are reusable.

### Phase 2: General tiled score computation — complete

Generate ISA loops over query/key tiles using computed tile counts.

Files:

- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`

Exit:

- multiple static shapes produce correct score tiles,
- padding tiles are handled correctly,
- `MMA` is called with fragment-staged Q/K tiles.

### Phase 3: Causal masking and softmax — complete

Implement:

- tile max,
- running max,
- exp via `EXP2`,
- running sum,
- score normalization.

Files:

- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`
- `tests/mlir/mgpu_to_isa_attention_tiled_softmax.mlir`
- `tests/operator_e2e/test_tiled_attention_kernels_e2e.py`

Exit:

- masked softmax matches a NumPy reference for `seq=8`, `head_dim=64`,
- masked softmax matches for `seq=32`, `head_dim=32`,
- generated ISA contains `softmax_row_loop`, `softmax_max_loop`,
  `softmax_sum_loop`, `softmax_store_loop`, `EXP2`, and `BAR.SYNC`.

The implementation uses a two-pass softmax over all key positions for each
query row, then writes normalized weights back into the score tiles.

### Phase 4: General tiled value accumulation — complete

Generate value-tile loops over computed `head_dim` tile count:

```text
output += normalized_score @ V_tile
```

Files:

- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`
- `tests/mlir/mgpu_to_isa_attention_tiled_full.mlir`
- `tests/operator_e2e/test_tiled_attention_kernels_e2e.py`

Exit:

- full attention matches a NumPy reference for `seq=8`, `head_dim=64`,
- full attention matches for `seq=32`, `head_dim=32`,
- `head_dim=20` verifies padded V tiles,
- generated ISA contains `value_q_tile_loop`, `value_v_tile_loop`,
  `value_k_tile_loop`, and `STG`.

The value stage loads normalized score tiles from shared memory, loads V
tiles from global memory, accumulates with `MMA`, and stores only in-bounds
output elements so padded rows/columns do not corrupt adjacent buffers.

### Phase 5: General loop and grid mapping

Map:

```text
batch * heads * query_tiles -> blocks
```

Use the existing wave scheduler.

Exit:

- arbitrary static batch/head/seq combinations run in multiple blocks,
- repeated launches are deterministic.

### Phase 6: Differential and lit tests

Add:

- mgpu-vs-software attention differential test,
- mgpu-vs-PyTorch attention test,
- FileCheck test ensuring `MMA`, loops, and `BAR.SYNC`.

Files:

- `tests/operator_e2e/test_attention_kernels_e2e.py`
- `tests/operator_e2e/test_software_kernels_e2e.py`
- `tests/mlir/mgpu_to_isa_attention.mlir`

Exit:

- all attention tests pass,
- pre-commit, unittest, and lit are green.

## 5. Suggested commit order

```text
1. fragment staging helpers — complete
2. tiled score computation — complete
3. causal masking and softmax — complete
4. tiled value accumulation — complete
5. loop and grid mapping
6. differential and lit tests
```

## 6. Risks

- Fragment register allocation may exceed 64 registers if staging is
  aggressive. Keep register count bounded and reuse fragments.
- Shared-memory offsets must be precomputed deterministically.
- Causal masking with tiled softmax requires careful handling of padding and
  `-inf`.
- MMA fragment layout must match the existing simulator expectations.
