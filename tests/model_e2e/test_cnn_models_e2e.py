import unittest

import numpy as np

from compiler.codegen import lower_kernel

from examples.cnn import detector_head, fpn_neck, full_detector, resnet_block
from tests.operator_e2e.helpers import make_device, run_operator_e2e


class TestCNNModelExamples(unittest.TestCase):
    """Step 016 CNN/object-detection model examples."""

    def test_resnet_block(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(1, 2, 4, 4)).astype(np.float32)
        w1 = rng.normal(size=(2, 2, 3, 3)).astype(np.float32)
        b1 = rng.normal(size=(2,)).astype(np.float32)
        w2 = rng.normal(size=(2, 2, 3, 3)).astype(np.float32)
        b2 = rng.normal(size=(2,)).astype(np.float32)

        graph = resnet_block.build_graph()
        result = run_operator_e2e(
            make_device(),
            graph,
            {"x": x, "w1": w1, "b1": b1, "w2": w2, "b2": b2},
            [resnet_block.reference(x, w1, b1, w2, b2)],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertIn('"conv2d"', graph.format())

    def test_fpn_neck(self):
        rng = np.random.default_rng(2)
        p3 = rng.normal(size=(1, 2, 4, 4)).astype(np.float32)
        p4 = rng.normal(size=(1, 2, 2, 2)).astype(np.float32)
        lateral_w = rng.normal(size=(2, 2, 1, 1)).astype(np.float32)
        proj_w = rng.normal(size=(2, 2, 1, 1)).astype(np.float32)

        graph = fpn_neck.build_graph()
        result = run_operator_e2e(
            make_device(),
            graph,
            {
                "p3": p3,
                "p4": p4,
                "lateral_w": lateral_w,
                "proj_w": proj_w,
            },
            [fpn_neck.reference(p3, p4, lateral_w, proj_w)],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertIn('"resize"', graph.format())

    def test_detector_head(self):
        boxes = np.array(
            [
                [10.0, 20.0, 4.0, 6.0],
                [30.0, 40.0, 2.0, 8.0],
                [50.0, 60.0, 3.0, 3.0],
            ],
            dtype=np.float32,
        )
        deltas = np.zeros_like(boxes)
        scores = np.array([0.9, 0.5, 0.8], dtype=np.float32)

        graph = detector_head.build_graph()
        result = run_operator_e2e(
            make_device(),
            graph,
            {"boxes": boxes, "deltas": deltas, "scores": scores},
            [detector_head.reference(boxes, deltas, scores)],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertIn('"nms"', graph.format())

    def test_full_detector(self):
        rng = np.random.default_rng(3)
        image = rng.normal(size=(1, 1, 4, 4)).astype(np.float32)
        conv_w = rng.normal(size=(1, 1, 3, 3)).astype(np.float32)
        boxes = np.array(
            [[0.0, 0.0, 2.0, 2.0], [5.0, 5.0, 2.0, 2.0]],
            dtype=np.float32,
        )
        deltas = np.zeros_like(boxes)
        scores = np.array([0.8, 0.7], dtype=np.float32)

        graph = full_detector.build_graph()
        expected = full_detector.reference(image, conv_w, boxes, deltas, scores)
        result = run_operator_e2e(
            make_device(),
            graph,
            {
                "image": image,
                "conv_w": conv_w,
                "boxes": boxes,
                "deltas": deltas,
                "scores": scores,
            },
            expected,
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertEqual(len(result.outputs), 2)


if __name__ == "__main__":
    unittest.main()
