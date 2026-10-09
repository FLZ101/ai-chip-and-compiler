"""ResNet-style block tests for the real-model stage.

Phase 6.1 keeps the model small enough to run in the deterministic simulator
while exercising the production frontends:

* PyTorch FX is lowered through ``TorchFXFrontend``.
* ONNX is lowered through ``ONNXFrontend``.

Both paths then use the shared C++ ``mgpu-opt`` backend via
``lower_kernel`` and compare against the native PyTorch/ONNX reference.
"""

from __future__ import annotations

import unittest

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

from compiler.codegen import lower_kernel
from compiler.frontends.onnx_frontend import ONNXFrontend
from compiler.frontends.torch_fx_frontend import TorchFXFrontend
from compiler.model_ir import DType

from tests.operator_e2e.helpers import make_device, run_operator_e2e


class ResNetBlock(nn.Module):
    """Two padded conv layers followed by a residual add."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 2, 3, padding=1)
        self.conv2 = nn.Conv2d(2, 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(torch.relu(self.conv1(x)))


def _make_module() -> ResNetBlock:
    module = ResNetBlock()
    rng = np.random.default_rng(12)
    with torch.no_grad():
        module.conv1.weight.copy_(
            torch.from_numpy(rng.normal(size=(2, 2, 3, 3)).astype(np.float32))
        )
        module.conv1.bias.copy_(
            torch.from_numpy(rng.normal(size=(2,)).astype(np.float32))
        )
        module.conv2.weight.copy_(
            torch.from_numpy(rng.normal(size=(2, 2, 3, 3)).astype(np.float32))
        )
        module.conv2.bias.copy_(
            torch.from_numpy(rng.normal(size=(2,)).astype(np.float32))
        )
    return module


def _example_input() -> np.ndarray:
    return np.arange(32, dtype=np.float32).reshape(1, 2, 4, 4) / 10.0


def _pytorch_inputs(module: ResNetBlock) -> dict[str, np.ndarray]:
    return {
        "x": _example_input(),
        "conv1_weight": module.conv1.weight.detach().numpy(),
        "conv1_bias": module.conv1.bias.detach().numpy(),
        "conv2_weight": module.conv2.weight.detach().numpy(),
        "conv2_bias": module.conv2.bias.detach().numpy(),
    }


def _build_onnx(module: ResNetBlock) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 4, 4])
    initializers = [
        helper.make_tensor(
            "w1",
            TensorProto.FLOAT,
            module.conv1.weight.shape,
            module.conv1.weight.detach().numpy().reshape(-1).tolist(),
        ),
        helper.make_tensor(
            "b1",
            TensorProto.FLOAT,
            module.conv1.bias.shape,
            module.conv1.bias.detach().numpy().tolist(),
        ),
        helper.make_tensor(
            "w2",
            TensorProto.FLOAT,
            module.conv2.weight.shape,
            module.conv2.weight.detach().numpy().reshape(-1).tolist(),
        ),
        helper.make_tensor(
            "b2",
            TensorProto.FLOAT,
            module.conv2.bias.shape,
            module.conv2.bias.detach().numpy().tolist(),
        ),
    ]
    nodes = [
        helper.make_node(
            "Conv",
            ["x", "w1", "b1"],
            ["c1"],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        helper.make_node("Relu", ["c1"], ["relu"]),
        helper.make_node(
            "Conv",
            ["relu", "w2", "b2"],
            ["c2"],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        helper.make_node("Add", ["x", "c2"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes, "resnet_block", [x], [output], initializer=initializers
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


class TestRealCNNResNetBlock(unittest.TestCase):
    def test_pytorch_resnet_block_runs_in_simulator(self):
        module = _make_module()
        graph_module = torch.fx.symbolic_trace(module)
        frontend = TorchFXFrontend(
            "pytorch_resnet_block",
            input_shapes={"x": ((1, 2, 4, 4), DType.F32)},
        )
        graph = frontend.convert(graph_module)
        inputs = _pytorch_inputs(module)

        with torch.no_grad():
            expected = module(torch.from_numpy(inputs["x"])).numpy()

        result = run_operator_e2e(
            make_device(),
            graph,
            inputs,
            [expected],
            lowering=lower_kernel,
            atol=1e-5,
        )
        np.testing.assert_allclose(result.outputs[0], expected, atol=1e-5)

    def test_onnx_resnet_block_runs_in_simulator(self):
        module = _make_module()
        model = _build_onnx(module)
        frontend = ONNXFrontend("onnx_resnet_block")
        graph = frontend.convert(model)

        x = _example_input()
        inputs = {
            "x": x,
            "w1": module.conv1.weight.detach().numpy(),
            "b1": module.conv1.bias.detach().numpy(),
            "w2": module.conv2.weight.detach().numpy(),
            "b2": module.conv2.bias.detach().numpy(),
        }
        expected = ReferenceEvaluator(model).run(None, {"x": x})[0]

        result = run_operator_e2e(
            make_device(),
            graph,
            inputs,
            [expected],
            lowering=lower_kernel,
            atol=1e-5,
        )
        np.testing.assert_allclose(result.outputs[0], expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
