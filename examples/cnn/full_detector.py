"""Small full CNN detector on a synthetic image."""

from __future__ import annotations

import numpy as np

from compiler.frontends.manual_frontend import ManualFrontend
from compiler.model_ir import DType, Graph

from .detector_head import reference as detector_reference


def build_graph() -> Graph:
    frontend = ManualFrontend("full_detector")
    image = frontend.add_input("image", DType.F32, (1, 1, 4, 4))
    conv_w = frontend.add_input("conv_w", DType.F32, (1, 1, 3, 3))
    boxes = frontend.add_input("boxes", DType.F32, (2, 4))
    deltas = frontend.add_input("deltas", DType.F32, (2, 4))
    scores = frontend.add_input("scores", DType.F32, (2,))

    feature = frontend.relu(frontend.conv2d(image, conv_w))
    feature = frontend.max_pool2d(feature, kernel=2, strides=2)
    feature = frontend.reshape(feature, (1, -1))

    decoded = frontend.box_decode(boxes, deltas)
    kept = frontend.nms(decoded, scores, iou_threshold=0.5)
    frontend.output(feature, kept)
    return frontend.build()


def reference(image, conv_w, boxes, deltas, scores):
    n, c, h, w = image.shape
    f, _, kh, kw = conv_w.shape
    oh = h - kh + 1
    ow = w - kw + 1
    conv = np.zeros((n, f, oh, ow), dtype=np.float32)
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
                                    * conv_w[ff, cc, ky, kx]
                                )
                    conv[nn, ff, oy, ox] = acc
    feature = np.maximum(conv, 0.0)
    feature = np.max(feature, axis=(2, 3), keepdims=True).reshape(1, -1)
    kept = detector_reference(boxes, deltas, scores)
    return [feature, kept]
