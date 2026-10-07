import unittest

import torch
import torch.nn as nn

from compiler.frontends.torch_fx_frontend import TorchFXFrontend
from compiler.mlir import MLIRBuilder
from compiler.model_ir import DType


class TinyTransformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        y = self.norm(x)
        y = self.proj(y)
        return torch.relu(y)


class TestTorchFXFrontend(unittest.TestCase):
    def test_transformer_block_exports_and_lowers(self):
        module = TinyTransformerBlock(4)
        graph_module = torch.fx.symbolic_trace(module)

        frontend = TorchFXFrontend(
            "fx_transformer",
            input_shapes={"x": ((2, 4), DType.F32)},
        )
        graph = frontend.convert(graph_module)
        graph.validate()

        mlir = MLIRBuilder().build(graph)
        self.assertIn('"layer_norm"', graph.format())
        self.assertIn('"linear"', graph.format())
        self.assertIn('"relu"', graph.format())
        self.assertIn("linalg.generic", mlir)


if __name__ == "__main__":
    unittest.main()
