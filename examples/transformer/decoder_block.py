"""Small GPT/decoder-style block with a causal mask."""

from __future__ import annotations

import numpy as np

from compiler.frontends.manual_frontend import ManualFrontend
from compiler.model_ir import DType, Graph


def build_graph() -> Graph:
    frontend = ManualFrontend("decoder_block")
    q = frontend.add_input("q", DType.F32, (1, 1, 2, 2))
    k = frontend.add_input("k", DType.F32, (1, 1, 2, 2))
    v = frontend.add_input("v", DType.F32, (1, 1, 2, 2))

    mask = frontend.causal_mask(query_len=2, key_len=2, batch=1)
    attn = frontend.masked_attention(q, k, v, mask)
    frontend.output(frontend.add(q, attn))
    return frontend.build()


def reference(q, k, v):
    mask = np.tril(np.ones((1, 1, 2, 2), dtype=np.float32))
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    scores = np.where(mask == 0, -np.inf, scores)
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    weights = exp / np.sum(exp, axis=-1, keepdims=True)
    attn = np.matmul(weights, v)
    return q + attn
