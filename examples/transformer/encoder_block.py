"""Small Transformer-style encoder block example."""

from __future__ import annotations

import numpy as np

from compiler.frontends.manual_frontend import ManualFrontend
from compiler.model_ir import DType, Graph


def build_graph() -> Graph:
    frontend = ManualFrontend("encoder_block")
    x = frontend.add_input("x", DType.F32, (2, 4))
    norm_w = frontend.add_input("norm_w", DType.F32, (4,))
    norm_b = frontend.add_input("norm_b", DType.F32, (4,))

    y = frontend.layer_norm(x, norm_w, norm_b)
    y = frontend.relu(y)
    frontend.output(frontend.add(x, y))
    return frontend.build()


def reference(x, norm_w, norm_b, eps=1e-5):
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    y = (x - mean) / np.sqrt(var + eps) * norm_w + norm_b
    return x + np.maximum(y, 0.0)
