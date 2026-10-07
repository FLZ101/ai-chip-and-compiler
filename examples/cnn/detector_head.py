"""Small YOLO/SSD-style detection head."""

from __future__ import annotations

import numpy as np

from compiler.frontends.manual_frontend import ManualFrontend
from compiler.model_ir import DType, Graph


def build_graph() -> Graph:
    frontend = ManualFrontend("detector_head")
    boxes = frontend.add_input("boxes", DType.F32, (3, 4))
    deltas = frontend.add_input("deltas", DType.F32, (3, 4))
    scores = frontend.add_input("scores", DType.F32, (3,))

    decoded = frontend.box_decode(boxes, deltas)
    frontend.output(frontend.nms(decoded, scores, iou_threshold=0.5))
    return frontend.build()


def reference(boxes, deltas, scores):
    cx = boxes[:, 0] + deltas[:, 0] * boxes[:, 2]
    cy = boxes[:, 1] + deltas[:, 1] * boxes[:, 3]
    w = boxes[:, 2] * np.exp(deltas[:, 2])
    h = boxes[:, 3] * np.exp(deltas[:, 3])
    decoded = np.stack(
        [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h],
        axis=1,
    ).astype(np.float32)

    def iou(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
        union = max(
            (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter,
            1.0,
        )
        return inter / union

    keep = np.ones(len(decoded), dtype=np.float32)
    for i in range(len(decoded)):
        for j in range(len(decoded)):
            if scores[j] > scores[i] and iou(decoded[i], decoded[j]) > 0.5:
                keep[i] = 0.0
    return np.where(keep, np.arange(len(decoded), dtype=np.float32), -1.0).astype(
        np.float32
    )
