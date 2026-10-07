import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper

from compiler.codegen import lower_kernel
from compiler.frontends.onnx_frontend import ONNXFrontend
from compiler.mlir import MLIRBuilder

from ..operator_e2e.helpers import make_device, run_operator_e2e


def _build_cnn_onnx():
    conv_w = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
    initializer = helper.make_tensor(
        "conv_w", TensorProto.FLOAT, conv_w.shape, conv_w.reshape(-1).tolist()
    )
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, 1, 1])
    nodes = [
        helper.make_node(
            "Conv",
            ["x", "conv_w"],
            ["conv_out"],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            dilations=[1, 1],
            group=1,
        ),
        helper.make_node("Relu", ["conv_out"], ["relu_out"]),
        helper.make_node(
            "MaxPool",
            ["relu_out"],
            ["out"],
            kernel_shape=[2, 2],
            strides=[2, 2],
        ),
    ]
    graph = helper.make_graph(nodes, "cnn", [x], [y], initializer=[initializer])
    model = helper.make_model(graph)
    model.ir_version = 8
    return model, conv_w


def _cnn_reference(x, weight):
    out_h = x.shape[2] - weight.shape[2] + 1
    out_w = x.shape[3] - weight.shape[3] + 1
    conv = np.zeros((1, 1, out_h, out_w), dtype=np.float32)
    for oy in range(out_h):
        for ox in range(out_w):
            acc = 0.0
            for ky in range(weight.shape[2]):
                for kx in range(weight.shape[3]):
                    acc += x[0, 0, oy + ky, ox + kx] * weight[0, 0, ky, kx]
            conv[0, 0, oy, ox] = acc
    relu = np.maximum(conv, 0.0)
    return np.max(relu[0, 0, 0:2, 0:2]).reshape(1, 1, 1, 1)


class TestONNXFrontend(unittest.TestCase):
    def test_onnx_cnn_imports_and_lowers(self):
        model, conv_w = _build_cnn_onnx()
        graph = ONNXFrontend("onnx_cnn").convert(model)
        graph.validate()
        mlir = MLIRBuilder().build(graph)
        self.assertIn("linalg.generic", mlir)

        x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        result = run_operator_e2e(
            make_device(),
            graph,
            {"x": x, "conv_w": conv_w},
            [_cnn_reference(x, conv_w)],
            lowering=lower_kernel,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            result.outputs[0], _cnn_reference(x, conv_w), atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
