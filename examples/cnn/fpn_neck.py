"""Small FPN-style top-down feature neck."""

from __future__ import annotations

import numpy as np

from compiler.frontends.manual_frontend import ManualFrontend
from compiler.model_ir import DType, Graph


def build_graph() -> Graph:
    frontend = ManualFrontend("fpn_neck")
    p3 = frontend.add_input("p3", DType.F32, (1, 2, 4, 4))
    p4 = frontend.add_input("p4", DType.F32, (1, 2, 2, 2))
    lateral_w = frontend.add_input("lateral_w", DType.F32, (2, 2, 1, 1))
    proj_w = frontend.add_input("proj_w", DType.F32, (2, 2, 1, 1))

    lateral = frontend.relu(frontend.conv2d(p4, lateral_w))
    upsampled = frontend.resize(lateral, sizes=(1, 2, 4, 4))
    projected = frontend.conv2d(p3, proj_w)
    frontend.output(frontend.add(upsampled, projected))
    return frontend.build()


def reference(
    p3: np.ndarray,
    p4: np.ndarray,
    lateral_w: np.ndarray,
    proj_w: np.ndarray,
) -> np.ndarray:
    def conv2d(image, weight):
        n, c, h, w = image.shape
        f, _, kh, kw = weight.shape
        oh = h - kh + 1
        ow = w - kw + 1
        out = np.zeros((n, f, oh, ow), dtype=np.float32)
        for nn in range(n):
            for ff in range(f):
                for oy in range(oh):
                    for ox in range(ow):
                        acc = 0.0
                        for cc in range(c):
                            for ky in range(kh):
                                for kx in range(kw):
                                    acc += (
                                        image[nn, cc, oy + ky, ox + kx]
                                        * weight[ff, cc, ky, kx]
                                    )
                        out[nn, ff, oy, ox] = acc
        return out

    lateral = np.maximum(conv2d(p4, lateral_w), 0.0)
    upsampled = np.repeat(np.repeat(lateral, 2, axis=2), 2, axis=3)
    projected = conv2d(p3, proj_w)
    return upsampled + projected
