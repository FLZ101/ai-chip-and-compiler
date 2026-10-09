"""Exported YOLO/SSD-style detection-head model test."""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from compiler.codegen import lower_kernel
from compiler.frontends import TorchFXFrontend
from compiler.frontends import detection_functions
from compiler.model_ir import DType

from examples.cnn.detector_head import reference
from tests.operator_e2e.helpers import make_device, run_operator_e2e


class DetectionHead(nn.Module):
    def forward(self, boxes: torch.Tensor, deltas: torch.Tensor, scores: torch.Tensor):
        return detection_functions.nms(
            detection_functions.box_decode(boxes, deltas), scores, 0.5
        )


class TestRealDetectionHead(unittest.TestCase):
    def test_pytorch_detection_head_runs_in_simulator(self):
        module = DetectionHead()
        graph_module = torch.fx.symbolic_trace(module)
        frontend = TorchFXFrontend(
            "pytorch_detection_head",
            input_shapes={
                "boxes": ((3, 4), DType.F32),
                "deltas": ((3, 4), DType.F32),
                "scores": ((3,), DType.F32),
            },
        )
        graph = frontend.convert(graph_module)

        boxes = np.asarray(
            [
                [10.0, 20.0, 4.0, 6.0],
                [30.0, 40.0, 2.0, 8.0],
                [50.0, 60.0, 3.0, 3.0],
            ],
            dtype=np.float32,
        )
        deltas = np.zeros_like(boxes)
        scores = np.asarray([0.9, 0.5, 0.8], dtype=np.float32)
        inputs = {"boxes": boxes, "deltas": deltas, "scores": scores}

        result = run_operator_e2e(
            make_device(),
            graph,
            inputs,
            [reference(boxes, deltas, scores)],
            lowering=lower_kernel,
            atol=1e-5,
        )

        self.assertIn("mgpu.box_decode", result.mlir_text)
        self.assertIn("mgpu.nms", result.mlir_text)
        np.testing.assert_allclose(
            result.outputs[0], reference(boxes, deltas, scores), atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
