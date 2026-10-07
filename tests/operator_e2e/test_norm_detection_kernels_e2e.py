import unittest

import numpy as np

from compiler.codegen import lower_kernel
from compiler.frontends.manual_frontend import ManualFrontend
from compiler.model_ir import DType

from .helpers import make_device, run_operator_e2e


def _batch_norm_reference(x, scale, bias, mean, var, eps=1e-5):
    mean = np.asarray(mean, dtype=np.float32).reshape(1, -1, 1, 1)
    var = np.asarray(var, dtype=np.float32).reshape(1, -1, 1, 1)
    scale = np.asarray(scale, dtype=np.float32).reshape(1, -1, 1, 1)
    bias = np.asarray(bias, dtype=np.float32).reshape(1, -1, 1, 1)
    return (x - mean) / np.sqrt(var + eps) * scale + bias


def _group_norm_reference(x, weight, bias, groups, eps=1e-5):
    n, c, h, w = x.shape
    channels_per_group = c // groups
    output = np.zeros_like(x)
    for group in range(groups):
        start = group * channels_per_group
        end = start + channels_per_group
        group_x = x[:, start:end, :, :]
        mean = group_x.mean()
        var = group_x.var()
        output[:, start:end, :, :] = (group_x - mean) / np.sqrt(var + eps) * weight[
            start:end
        ].reshape(1, -1, 1, 1) + bias[start:end].reshape(1, -1, 1, 1)
    return output


def _box_decode_reference(boxes, deltas):
    boxes = np.asarray(boxes, dtype=np.float32)
    deltas = np.asarray(deltas, dtype=np.float32)
    cx = boxes[:, 0] + deltas[:, 0] * boxes[:, 2]
    cy = boxes[:, 1] + deltas[:, 1] * boxes[:, 3]
    w = boxes[:, 2] * np.exp(deltas[:, 2])
    h = boxes[:, 3] * np.exp(deltas[:, 3])
    return np.stack(
        [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h],
        axis=1,
    ).astype(np.float32)


def _iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = max(area_a + area_b - inter, 1.0)
    return inter / union


def _nms_reference(boxes, scores, threshold=0.5):
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    keep = np.ones(len(boxes), dtype=np.float32)
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if scores[j] > scores[i] and _iou(boxes[i], boxes[j]) > threshold:
                keep[i] = 0.0
    return np.where(keep, np.arange(len(boxes), dtype=np.float32), -1.0).astype(
        np.float32
    )


class TestNormAndDetectionKernels(unittest.TestCase):
    """Step 011 normalization and detection helper tests."""

    def test_batch_norm(self):
        x = np.arange(1 * 2 * 2 * 2, dtype=np.float32).reshape(1, 2, 2, 2)
        scale = np.array([1.0, 2.0], dtype=np.float32)
        bias = np.array([0.5, -0.5], dtype=np.float32)
        mean = np.array([1.0, 2.0], dtype=np.float32)
        var = np.array([0.5, 1.0], dtype=np.float32)

        frontend = ManualFrontend("batch_norm")
        value = frontend.add_input("x", DType.F32, x.shape)
        s = frontend.add_input("s", DType.F32, scale.shape)
        b = frontend.add_input("b", DType.F32, bias.shape)
        m = frontend.add_input("m", DType.F32, mean.shape)
        v = frontend.add_input("v", DType.F32, var.shape)
        frontend.output(frontend.batch_norm(value, s, b, m, v))

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"x": x, "s": scale, "b": bias, "m": mean, "v": var},
            [_batch_norm_reference(x, scale, bias, mean, var)],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertIn("RSQRT", result.isa_text)

    def test_group_norm(self):
        x = np.arange(1 * 2 * 2 * 2, dtype=np.float32).reshape(1, 2, 2, 2)
        weight = np.array([1.0, 2.0], dtype=np.float32)
        bias = np.array([0.0, 1.0], dtype=np.float32)

        frontend = ManualFrontend("group_norm")
        value = frontend.add_input("x", DType.F32, x.shape)
        w = frontend.add_input("w", DType.F32, weight.shape)
        b = frontend.add_input("b", DType.F32, bias.shape)
        frontend.output(frontend.group_norm(value, w, b, num_groups=2))

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"x": x, "w": weight, "b": bias},
            [_group_norm_reference(x, weight, bias, groups=2)],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertIn("RSQRT", result.isa_text)

    def test_box_decode(self):
        boxes = np.array(
            [[10.0, 20.0, 4.0, 6.0], [30.0, 40.0, 2.0, 8.0]],
            dtype=np.float32,
        )
        deltas = np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.1, -0.1, 0.2, -0.2]],
            dtype=np.float32,
        )

        frontend = ManualFrontend("box_decode")
        b = frontend.add_input("boxes", DType.F32, boxes.shape)
        d = frontend.add_input("deltas", DType.F32, deltas.shape)
        frontend.output(frontend.box_decode(b, d))

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"boxes": boxes, "deltas": deltas},
            [_box_decode_reference(boxes, deltas)],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertIn("EXP2", result.isa_text)

    def test_nms(self):
        boxes = np.array(
            [[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0], [5.0, 5.0, 7.0, 7.0]],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        frontend = ManualFrontend("nms")
        b = frontend.add_input("boxes", DType.F32, boxes.shape)
        s = frontend.add_input("scores", DType.F32, scores.shape)
        frontend.output(frontend.nms(b, s, iou_threshold=0.5))

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"boxes": boxes, "scores": scores},
            [_nms_reference(boxes, scores, threshold=0.5)],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertIn("CMP.GT", result.isa_text)

    def test_small_detector_head_graph(self):
        boxes = np.array(
            [[10.0, 20.0, 4.0, 6.0], [30.0, 40.0, 2.0, 8.0]],
            dtype=np.float32,
        )
        deltas = np.zeros_like(boxes)
        scores = np.array([0.9, 0.5], dtype=np.float32)

        frontend = ManualFrontend("detector_head")
        b = frontend.add_input("boxes", DType.F32, boxes.shape)
        d = frontend.add_input("deltas", DType.F32, deltas.shape)
        s = frontend.add_input("scores", DType.F32, scores.shape)
        decoded = frontend.box_decode(b, d)
        frontend.output(frontend.nms(decoded, s, iou_threshold=0.5))

        decoded_ref = _box_decode_reference(boxes, deltas)
        expected = _nms_reference(decoded_ref, scores, threshold=0.5)
        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"boxes": boxes, "deltas": deltas, "scores": scores},
            [expected],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertGreater(len(result.isa_text.splitlines()), 10)


if __name__ == "__main__":
    unittest.main()
