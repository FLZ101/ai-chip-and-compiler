import unittest

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper

from compiler.mlir import MLIRCanonicalizer, ONNXImporter, TorchMLIRImporter


class TinyLinearBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x):
        return torch.relu(self.proj(x))


def _build_cnn_onnx() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 8, 8])
    weight = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [1, 1, 3, 3],
        np.ones((1, 1, 3, 3), dtype=np.float32).reshape(-1).tolist(),
    )
    bias = helper.make_tensor(
        "bias",
        TensorProto.FLOAT,
        [1],
        np.zeros((1,), dtype=np.float32).tolist(),
    )
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 3, 3])
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["x", "weight", "bias"],
                ["conv"],
                kernel_shape=[3, 3],
            ),
            helper.make_node("Relu", ["conv"], ["relu"]),
            helper.make_node(
                "MaxPool",
                ["relu"],
                ["output"],
                kernel_shape=[2, 2],
                strides=[2, 2],
            ),
        ],
        "cnn",
        [x],
        [output],
        initializer=[weight, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def _build_qdq_onnx() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])
    scale = helper.make_tensor(
        "scale", TensorProto.FLOAT, [], np.asarray(0.5, dtype=np.float32).tolist()
    )
    zero_point = helper.make_tensor("zero_point", TensorProto.INT8, [], [0])
    graph = helper.make_graph(
        [
            helper.make_node(
                "QuantizeLinear", ["x", "scale", "zero_point"], ["quantized"]
            ),
            helper.make_node(
                "DequantizeLinear", ["quantized", "scale", "zero_point"], ["output"]
            ),
        ],
        "qdq",
        [x],
        [output],
        initializer=[scale, zero_point],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


class TestMLIRCanonicalizer(unittest.TestCase):
    def test_fx_import_canonicalizes_to_linalg(self):
        model = TinyLinearBlock()
        imported = TorchMLIRImporter().import_module(
            model,
            torch.zeros(2, 4),
            func_name="linear_block",
        )

        result = MLIRCanonicalizer().canonicalize(imported)

        self.assertEqual(result.source, "fx")
        self.assertEqual(result.target, "linalg-on-tensors")
        self.assertIn("linalg.generic", result.text)
        self.assertIn("arith.", result.text)
        self.assertNotIn("torch.aten", result.text)
        self.assertTrue(result.mlir_module.operation.verify())

    def test_fx_canonicalization_is_deterministic(self):
        model = TinyLinearBlock()
        example = torch.zeros(2, 4)
        first = TorchMLIRImporter().import_module(
            model,
            example,
            func_name="linear_block",
        )
        second = TorchMLIRImporter().import_module(
            model,
            example,
            func_name="linear_block",
        )

        first_canonical = MLIRCanonicalizer().canonicalize(first)
        second_canonical = MLIRCanonicalizer().canonicalize(second)

        self.assertEqual(first_canonical.text, second_canonical.text)

    def test_onnx_import_canonicalizes_to_linalg(self):
        imported = ONNXImporter().import_model(_build_cnn_onnx())

        result = MLIRCanonicalizer().canonicalize(imported)

        self.assertEqual(result.source, "onnx")
        self.assertEqual(result.target, "linalg-on-tensors")
        self.assertIn("linalg.conv_2d_nchw_fchw", result.text)
        self.assertIn("linalg.pooling_nchw_max", result.text)
        self.assertIn("arith.", result.text)
        self.assertNotIn("torch.operator", result.text)
        self.assertTrue(result.mlir_module.operation.verify())

    def test_onnx_import_canonicalizes_to_tosa(self):
        imported = ONNXImporter().import_model(_build_cnn_onnx())

        result = MLIRCanonicalizer().canonicalize(imported, target="tosa")

        self.assertEqual(result.target, "tosa")
        self.assertIn("tosa.", result.text)
        self.assertNotIn("torch.operator", result.text)
        self.assertTrue(result.mlir_module.operation.verify())

    def test_onnx_qdq_canonicalizes_to_arith_conversions(self):
        imported = ONNXImporter().import_model(_build_qdq_onnx())

        result = MLIRCanonicalizer().canonicalize(imported)

        self.assertEqual(result.target, "linalg-on-tensors")
        self.assertIn("linalg.generic", result.text)
        self.assertIn("arith.fptosi", result.text)
        self.assertIn("arith.sitofp", result.text)
        self.assertNotIn("onnx.QuantizeLinear", result.text)
        self.assertNotIn("onnx.DequantizeLinear", result.text)
        self.assertTrue(result.mlir_module.operation.verify())


if __name__ == "__main__":
    unittest.main()
