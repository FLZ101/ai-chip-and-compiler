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

RESNET18_WEIGHTS = (
    Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
)


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
        )

        self.assertIn("mgpu.conv2d", result.mlir_text)
        self.assertIn("mgpu.batch_norm", result.mlir_text)
        np.testing.assert_allclose(result.outputs[0], expected, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
