# Tiled MMA Attention Plan

## 1. Goal

Replace the current small-shape `mgpu.attention` MMA path with a tiled,
loop-based implementation that supports:

```text
batch=1
heads=12
seq=8
head_dim=64
```

and later larger sequences.

The generated ISA must remain bounded and must match PyTorch on GPT-2-like
shapes.

## 2. Current limitation

`emitAttentionIsa` returns unsupported for:

```text
seq > 16
head_dim > 16
```

It assumes a single 16x16 MMA and unrolled fragments. GPT-2 head dim is 64,
so this path cannot lower the real model.

## 3. Design

### 3.1 Tile sizes

- Query tile: `16 x 16`
- Key/value tile: `16 x 16`
- Output tile: `16 x 16`

For `head_dim=64`, four key/value tiles are needed.

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

## 4. Implementation phases

### Phase 1: Fragment staging helpers

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

### Phase 2: Tiled score computation

Generate ISA loops over query/key tiles.

Files:

- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`

Exit:

- `seq=8`, `head_dim=64` score tiles are produced,
- `MMA` is called with fragment-staged Q/K tiles.

### Phase 3: Causal masking and softmax

Implement:

- tile max,
- running max,
- exp via `EXP2`,
- running sum,
- score normalization.

Files:

- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`

Exit:

- causal attention matches a NumPy softmax reference on cropped shapes.

### Phase 4: Tiled value accumulation

Generate value-tile loops:

```text
output += normalized_score @ V_tile
```

Files:

- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`

Exit:

- full attention output matches PyTorch for `heads=12`, `seq=8`, `head_dim=64`.

### Phase 5: Loop and grid mapping

Map:

```text
batch * heads * query_tiles -> blocks
```

Use the existing wave scheduler.

Exit:

- large attention runs in multiple blocks,
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
1. fragment staging helpers
2. tiled score computation
3. causal masking and softmax
4. tiled value accumulation
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
