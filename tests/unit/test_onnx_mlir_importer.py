import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper

from compiler.frontends.onnx_frontend import ONNXFrontend
from compiler.mlir import ONNXImporter


def _build_cnn_onnx() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 8, 8])
    weight = helper.make_tensor(
        "conv_weight",
        TensorProto.FLOAT,
        [1, 1, 3, 3],
        np.ones((1, 1, 3, 3), dtype=np.float32).reshape(-1).tolist(),
    )
    bias = helper.make_tensor(
        "conv_bias",
        TensorProto.FLOAT,
        [1],
        np.zeros((1,), dtype=np.float32).tolist(),
    )
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 3, 3])
    nodes = [
        helper.make_node(
            "Conv",
            ["x", "conv_weight", "conv_bias"],
            ["conv_out"],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
        ),
        helper.make_node("Relu", ["conv_out"], ["relu_out"]),
        helper.make_node(
            "MaxPool",
            ["relu_out"],
            ["output"],
            kernel_shape=[2, 2],
            strides=[2, 2],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "cnn",
        [x],
        [output],
        initializer=[weight, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def _build_transformer_block_onnx() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])
    initializers = [
        helper.make_tensor(
            "ln_weight",
            TensorProto.FLOAT,
            [4],
            np.ones(4, dtype=np.float32).tolist(),
        ),
        helper.make_tensor(
            "ln_bias",
            TensorProto.FLOAT,
            [4],
            np.zeros(4, dtype=np.float32).tolist(),
        ),
        helper.make_tensor(
            "proj_weight",
            TensorProto.FLOAT,
            [4, 4],
            np.eye(4, dtype=np.float32).reshape(-1).tolist(),
        ),
        helper.make_tensor(
            "proj_bias",
            TensorProto.FLOAT,
            [4],
            np.zeros(4, dtype=np.float32).tolist(),
        ),
    ]
    nodes = [
        helper.make_node(
            "LayerNormalization",
            ["x", "ln_weight", "ln_bias"],
            ["norm_out"],
            axis=-1,
            epsilon=1e-5,
        ),
        helper.make_node("MatMul", ["norm_out", "proj_weight"], ["proj_out"]),
        helper.make_node("Add", ["proj_out", "proj_bias"], ["add_out"]),
        helper.make_node("Relu", ["add_out"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "transformer_block",
        [x],
        [output],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


class TestONNXMLIRImporter(unittest.TestCase):
    def test_onnx_cnn_imports_without_manual_mapping(self):
        model = _build_cnn_onnx()

        result = ONNXImporter().import_model(model)

        self.assertIn("func.func @cnn", result.text)
        self.assertIn('"onnx.Conv"', result.text)
        self.assertIn('"onnx.Relu"', result.text)
        self.assertIn('"onnx.MaxPool"', result.text)
        self.assertEqual(result.initializer_names, ("conv_weight", "conv_bias"))
        self.assertEqual(result.dynamic_inputs, {"x": (1, 1, 8, 8)})
        self.assertTrue(result.mlir_module.operation.verify())

    def test_onnx_dynamic_dimension_is_preserved(self):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])
        weight = helper.make_tensor(
            "weight",
            TensorProto.FLOAT,
            [4, 4],
            np.eye(4, dtype=np.float32).reshape(-1).tolist(),
        )
        output = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ["batch", 4]
        )
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["x", "weight"], ["output"])],
            "dynamic_matmul",
            [x],
            [output],
            initializer=[weight],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 13)],
        )
        model.ir_version = 8

        result = ONNXImporter().import_model(model)

        self.assertEqual(result.dynamic_inputs, {"x": ("batch", 4)})
        self.assertIn("!torch.vtensor<[?,4]", result.text)
        self.assertTrue(result.mlir_module.operation.verify())

    def test_onnx_transformer_block_imports(self):
        model = _build_transformer_block_onnx()

        result = ONNXFrontend("onnx_transformer").import_mlir(model)

        self.assertIn("func.func @transformer_block", result.text)
        self.assertIn('"onnx.LayerNormalization"', result.text)
        self.assertIn('"onnx.MatMul"', result.text)
        self.assertIn('"onnx.Add"', result.text)
        self.assertIn('"onnx.Relu"', result.text)
        self.assertEqual(
            result.initializer_names,
            ("ln_weight", "ln_bias", "proj_weight", "proj_bias"),
        )
        self.assertTrue(result.mlir_module.operation.verify())


if __name__ == "__main__":
    unittest.main()
