# GPT-2-Small Completion Steps

These are the remaining concrete steps, ordered, to finish
`docs/gpt2-small-run-plan.md`.

## Step 1: Add positional broadcast add — complete

Implement software broadcast add for token embeddings plus positional
embeddings.

Files:

- `tests/operator_e2e/helpers.py`
- `tests/operator_e2e/test_software_kernels_e2e.py`

Exit:

- `(1, seq, hidden) + (1, seq, hidden)` and
  `(1, seq, hidden) + (seq, hidden)` match NumPy broadcasting.

## Step 2: Add GPT-2 Model IR graph construction — complete

Build a `Graph` for GPT-2-small from PyTorch or a manually constructed subset.

Files:

- `compiler/frontends/torch_fx_frontend.py` or a new GPT-2-specific frontend
- `tests/model_e2e/test_real_llm_graph.py`

Exit:

- full GPT-2 graph builds,
- graph validation passes,
- output shape is `(1, seq, vocab)`.

Implemented as a full manual GPT-2 graph with `vocab=50257`, `hidden=768`,
`heads=12`, and `layers=12`. The graph is validated in
`tests/model_e2e/test_real_llm_graph.py`.

## Step 3: Complete multi-head attention software path — complete

Implement and test decomposed attention:

```text
q,k,v projections
  -> reshape/split into heads
  -> q @ k^T
  -> scale
  -> causal mask + softmax
  -> attn @ v
  -> merge heads
  -> output projection
```

Files:

- `tests/operator_e2e/test_software_kernels_e2e.py`
- `tests/operator_e2e/test_attention_kernels_e2e.py`

Exit:

- `heads=12`, `seq=8`, `head_dim=64` matches PyTorch.

## Step 4: Generalize mgpu attention — complete

Extend `emitAttentionIsa` to handle `heads=12`, `head_dim=64`, and bounded
loops instead of unrolled tiny shapes.

Files:

- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`
- `tests/mlir/mgpu_to_isa_attention.mlir`

Exit:

- mgpu attention matches software attention on a cropped equivalent shape,
- generated ISA size is bounded.

The mapped tiled attention kernel now handles arbitrary batch/head/query-tile
shapes and broadcast causal masks.

## Step 5: Add large GPT-2 linear tests — complete

Test software and mgpu linear for:

```text
768 -> 3072
3072 -> 768
768 -> vocab
```

Files:

- `tests/operator_e2e/test_software_kernels_e2e.py`
- `tests/operator_e2e/test_hardware_scaling_e2e.py`

Exit:

- outputs match PyTorch/NumPy,
- runtime is acceptable for `seq=8`.

Implemented:

- software execution for `8x768 -> 8x3072` and `8x3072 -> 8x768`,
- opt-in software execution for `8x768 -> 8x50257`,
- bounded `mgpu.linear` lowering checks for all three shapes.

## Step 6: Build GPT-2 memory plan — complete

Use the lifetime-aware planner for embeddings, all 12 block weights,
activations, logits, and KV cache.

Files:

- `runtime/memory_plan.py`
- `compiler/mlir/mgpu_runner.py`
- `tests/model_e2e/test_real_llm_memory.py`

Exit:

- peak memory fits configured global memory,
- plan is deterministic,
- planner reports total and peak bytes.

Implemented with explicit GPT-2-sized `PlannedBuffer` entries in
`tests/model_e2e/test_real_llm_memory.py`. Full canonicalized-module injection
now gets through bufferization after `tm_tensor.scan` lowering; the remaining
blocker is unsupported `arith.index_cast` lowering in `mgpu-to-isa`.

## Step 7: Implement greedy autoregressive loop

Run prompt encoding followed by repeated one-token decode steps.

Files:

- `runtime/driver.py`
- `examples/transformer/gpt2_decode.py`
- `tests/model_e2e/test_real_llm_decode.py`

Exit:

- generated token IDs match PyTorch greedy decode for several steps,
- KV cache grows without corrupting previous tokens.

## Step 8: Add numerical gates

Compare logits, attention, and generated token IDs with PyTorch.

Files:

- `tests/model_e2e/test_real_llm_decode.py`
- `docs/tolerances.md`

Exit:

- static `seq=8` forward matches PyTorch,
- greedy decode token IDs match,
- pre-commit, unittest, and lit are green.

## Suggested commit order

```text
1. broadcast add — complete
2. GPT-2 graph construction — complete
3. software attention — complete
4. mgpu attention — complete
5. large linear tests — complete
6. GPT-2 memory plan — complete
7. greedy decode loop
8. numerical gates
```
