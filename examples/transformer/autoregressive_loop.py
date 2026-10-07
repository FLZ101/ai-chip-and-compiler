"""Small two-step autoregressive decode loop example."""

from __future__ import annotations

import numpy as np

from compiler.frontends.manual_frontend import ManualFrontend
from compiler.model_ir import DType, Graph


def build_graph() -> Graph:
    frontend = ManualFrontend("autoregressive_loop")
    cache = frontend.add_input("cache", DType.F32, (1, 1, 4, 2))
    pos0 = frontend.add_input("pos0", DType.I64, (1, 1))
    pos1 = frontend.add_input("pos1", DType.I64, (1, 1))

    first = frontend.kv_cache_gather(cache, pos0)
    second = frontend.kv_cache_gather(cache, pos1)
    frontend.output(first, second)
    return frontend.build()


def reference(cache, pos0, pos1):
    p0 = int(pos0[0, 0])
    p1 = int(pos1[0, 0])
    return [cache[:, :, p0 : p0 + 1, :], cache[:, :, p1 : p1 + 1, :]]
