"""Execute a real pretrained ResNet18 stem in the simulator.

This is a cropped static-shape execution milestone for the hardware-scaling
plan.  The full 224x224 ResNet18 remains blocked by codegen scalability, but
this test proves that pretrained conv + batch-norm + ReLU weights can run
end to end through the current C++ backend.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from compiler.codegen import lower_kernel
from compiler.frontends import TorchFXFrontend
from compiler.model_ir import DType

from tests.operator_e2e.helpers import make_device, run_operator_e2e

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_RESNET18_WEIGHTS = (
    _PROJECT_ROOT / ".cache" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
)
_HOME_RESNET18_WEIGHTS = (
    Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
)
RESNET18_WEIGHTS = (
    _PROJECT_RESNET18_WEIGHTS
    if _PROJECT_RESNET18_WEIGHTS.exists()
    else _HOME_RESNET18_WEIGHTS
)


def _full_resnet18_inputs(model, graph, image):
    """Return the named tensor inputs consumed by the full FX graph."""
    inputs = {"x": image}
    for value in graph.inputs:
        if value.name == "x":
            continue
        if value.name.endswith("_weight"):
            path = value.name[: -len("_weight")]
            inputs[value.name] = model.get_parameter(path + ".weight").detach().numpy()
        elif value.name.endswith("_bias"):
            path = value.name[: -len("_bias")]
            inputs[value.name] = model.get_parameter(path + ".bias").detach().numpy()
        elif value.name.endswith("_running_mean"):
            path = value.name[: -len("_running_mean")]
            inputs[value.name] = (
                model.get_buffer(path + ".running_mean").detach().numpy()
            )
        elif value.name.endswith("_running_var"):
            path = value.name[: -len("_running_var")]
            inputs[value.name] = (
                model.get_buffer(path + ".running_var").detach().numpy()
            )
        else:
            raise AssertionError(f"unknown ResNet18 graph input: {value.name}")
    return inputs


class ResNet18Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn1(self.conv1(x)))


@unittest.skipUnless(
    RESNET18_WEIGHTS.exists() or os.environ.get("MGPU_DOWNLOAD_MODELS") == "1",
    "download real models with scripts/download_real_models.py or set MGPU_DOWNLOAD_MODELS=1",
)
class TestRealCNNExecution(unittest.TestCase):
    def test_pretrained_resnet18_stem_runs_in_simulator(self):
        base = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
        module = ResNet18Stem().eval()
        with torch.no_grad():
            module.conv1.weight.copy_(base.conv1.weight)
            module.bn1.load_state_dict(base.bn1.state_dict())

        graph_module = torch.fx.symbolic_trace(module)
        frontend = TorchFXFrontend(
            "resnet18_stem",
            input_shapes={"x": ((1, 3, 4, 4), DType.F32)},
        )
        graph = frontend.convert(graph_module)

        x = np.random.default_rng(0).normal(size=(1, 3, 4, 4)).astype(np.float32)
        with torch.no_grad():
            expected = module(torch.from_numpy(x)).numpy()
        inputs = {
            "x": x,
            "conv1_weight": module.conv1.weight.detach().numpy(),
            "bn1_weight": module.bn1.weight.detach().numpy(),
            "bn1_bias": module.bn1.bias.detach().numpy(),
            "bn1_running_mean": module.bn1.running_mean.detach().numpy(),
            "bn1_running_var": module.bn1.running_var.detach().numpy(),
        }

        result = run_operator_e2e(
            make_device(global_size=1 << 24, sm_count=16),
            graph,
            inputs,
            [expected],
            lowering=lower_kernel,
            atol=1e-4,
            profile=True,
        )

        self.assertIn("mgpu.conv2d", result.mlir_text)
        self.assertIn("mgpu.batch_norm", result.mlir_text)
        np.testing.assert_allclose(result.outputs[0], expected, atol=1e-4)
        self.assertGreater(result.cycles, 0)
        self.assertIsNotNone(result.profile)
        self.assertGreater(result.profile.issued_instructions, 0)

    def test_full_resnet18_graph_has_final_logits(self):
        model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
        graph_module = torch.fx.symbolic_trace(model)
        frontend = TorchFXFrontend(
            "resnet18",
            input_shapes={"x": ((1, 3, 4, 4), DType.F32)},
        )
        graph = frontend.convert(graph_module)

        self.assertEqual(len(graph.nodes), 70)
        self.assertEqual(graph.outputs[0].shape, (1, 1000))
        self.assertIn("fc_weight", {value.name for value in graph.inputs})
        self.assertIn("global_avg_pool", {node.op.value for node in graph.nodes})
        self.assertIn("linear", {node.op.value for node in graph.nodes})

    @unittest.skipUnless(
        os.environ.get("MGPU_FULL_RESNET18") == "1",
        "full ResNet18 simulator execution is opt-in and intentionally slow",
    )
    def test_full_resnet18_forward_runs_in_simulator(self):
        model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
        graph_module = torch.fx.symbolic_trace(model)
        frontend = TorchFXFrontend(
            "resnet18",
            input_shapes={"x": ((1, 3, 224, 224), DType.F32)},
        )
        graph = frontend.convert(graph_module)
        x = np.random.default_rng(0).normal(size=(1, 3, 224, 224)).astype(np.float32)
        inputs = _full_resnet18_inputs(model, graph, x)
        with torch.no_grad():
            expected = model(torch.from_numpy(x)).numpy()

        result = run_operator_e2e(
            make_device(global_size=1 << 27, sm_count=8),
            graph,
            inputs,
            [expected],
            lowering=lower_kernel,
            atol=5e-3,
            rtol=5e-3,
            profile=True,
            progress=True,
        )

        self.assertGreater(result.cycles, 0)
        self.assertIsNotNone(result.profile)
        np.testing.assert_allclose(
            result.outputs[0],
            expected,
            atol=5e-3,
            rtol=5e-3,
        )


if __name__ == "__main__":
    unittest.main()
