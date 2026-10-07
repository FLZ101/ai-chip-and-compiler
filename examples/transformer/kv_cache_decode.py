"""Small KV-cache decode-step example."""

from __future__ import annotations

import numpy as np

from compiler.frontends.manual_frontend import ManualFrontend
from compiler.model_ir import DType, Graph


def build_graph() -> Graph:
    frontend = ManualFrontend("kv_cache_decode")
    cache = frontend.add_input("cache", DType.F32, (1, 1, 4, 2))
    positions = frontend.add_input("positions", DType.I64, (1, 1))
    frontend.output(frontend.kv_cache_gather(cache, positions))
    return frontend.build()


def reference(cache, positions):
    pos = int(positions[0, 0])
    return cache[:, :, pos : pos + 1, :]
