"""Small ResNet-style residual block."""

from __future__ import annotations

import numpy as np

from compiler.frontends.manual_frontend import ManualFrontend
from compiler.model_ir import DType, Graph


def build_graph() -> Graph:
    frontend = ManualFrontend("resnet_block")
    x = frontend.add_input("x", DType.F32, (1, 2, 4, 4))
    w1 = frontend.add_input("w1", DType.F32, (2, 2, 3, 3))
    b1 = frontend.add_input("b1", DType.F32, (2,))
    w2 = frontend.add_input("w2", DType.F32, (2, 2, 3, 3))
    b2 = frontend.add_input("b2", DType.F32, (2,))

    y = frontend.relu(frontend.conv2d(x, w1, b1, padding=1))
    y = frontend.conv2d(y, w2, b2, padding=1)
    frontend.output(frontend.add(x, y))
    return frontend.build()


def reference(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    def conv2d(image, weight, bias, padding):
        n, c, h, w = image.shape
        f, _, kh, kw = weight.shape
        oh = h + 2 * padding - kh + 1
        ow = w + 2 * padding - kw + 1
        out = np.zeros((n, f, oh, ow), dtype=np.float32)
        for nn in range(n):
            for ff in range(f):
                for oy in range(oh):
                    for ox in range(ow):
                        acc = bias[ff]
                        for cc in range(c):
                            for ky in range(kh):
                                for kx in range(kw):
                                    iy = oy + ky - padding
                                    ix = ox + kx - padding
                                    if 0 <= iy < h and 0 <= ix < w:
                                        acc += (
                                            image[nn, cc, iy, ix]
                                            * weight[ff, cc, ky, kx]
                                        )
                        out[nn, ff, oy, ox] = acc
        return out

    y = conv2d(x, w1, b1, 1)
    y = np.maximum(y, 0.0)
    y = conv2d(y, w2, b2, 1)
    return x + y
