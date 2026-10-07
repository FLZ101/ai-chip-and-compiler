import unittest

import numpy as np
import torch
import torch.nn as nn

from compiler.codegen import lower_kernel
from compiler.frontends.torch_fx_frontend import TorchFXFrontend
from compiler.model_ir import DType

from tests.operator_e2e.helpers import make_device, run_operator_e2e


class TinyPyTorchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, bias=False)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(torch.relu(self.conv(x)))


class TestPyTorchModelE2E(unittest.TestCase):
    def test_traced_pytorch_cnn_runs_in_simulator(self):
        module = TinyPyTorchCNN()
        with torch.no_grad():
            module.conv.weight.copy_(
                torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]])
            )

        graph_module = torch.fx.symbolic_trace(module)
        frontend = TorchFXFrontend(
            "pytorch_cnn",
            input_shapes={"x": ((1, 1, 4, 4), DType.F32)},
        )
        graph = frontend.convert(graph_module)

        x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        weight = module.conv.weight.detach().numpy()
        with torch.no_grad():
            expected = module(torch.from_numpy(x)).detach().numpy()

        result = run_operator_e2e(
            make_device(),
            graph,
            {"x": x, "conv_weight": weight},
            [expected],
            lowering=lower_kernel,
            atol=1e-5,
        )
        np.testing.assert_allclose(result.outputs[0], expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
