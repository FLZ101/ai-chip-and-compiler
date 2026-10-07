# Step 017: Transformer/LLM Model Examples

This document describes the small Transformer/LLM model examples.

Step 017 adds runnable examples under [../examples/transformer](../examples/transformer):

- Transformer encoder block,
- GPT/decoder-style block with causal mask,
- KV-cache decode step,
- small autoregressive loop.

Each example builds a unified Model IR graph and has a NumPy golden reference.

---

## 1. Examples

### Encoder block

File: [../examples/transformer/encoder_block.py](../examples/transformer/encoder_block.py)

`LayerNorm -> ReLU -> residual Add`.

### Decoder block

File: [../examples/transformer/decoder_block.py](../examples/transformer/decoder_block.py)

`causal_mask -> MaskedAttention -> residual Add`.

### KV-cache decode

File: [../examples/transformer/kv_cache_decode.py](../examples/transformer/kv_cache_decode.py)

`KVCacheGather` from a rank-4 cache.

### Autoregressive loop

File: [../examples/transformer/autoregressive_loop.py](../examples/transformer/autoregressive_loop.py)

Two sequential `KVCacheGather` steps from the same cache.

---

## 2. Files changed

- [../examples/transformer/__init__.py](../examples/transformer/__init__.py)
- [../examples/transformer/encoder_block.py](../examples/transformer/encoder_block.py)
- [../examples/transformer/decoder_block.py](../examples/transformer/decoder_block.py)
- [../examples/transformer/kv_cache_decode.py](../examples/transformer/kv_cache_decode.py)
- [../examples/transformer/autoregressive_loop.py](../examples/transformer/autoregressive_loop.py)
- [../tests/model_e2e/test_transformer_models_e2e.py](../tests/model_e2e/test_transformer_models_e2e.py)

---

## 3. Tests

New model-level tests run all four examples through the shared E2E harness and
compare them with NumPy references.

Run:

```bash
source .venv/bin/activate
python -m unittest discover -v
```

All existing tests continue to pass.
