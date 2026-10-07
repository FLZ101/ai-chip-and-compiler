import unittest

import torch
import torch.nn as nn

from compiler.frontends.torch_fx_frontend import TorchFXFrontend
from compiler.mlir import TorchMLIRImporter
from compiler.model_ir import DType


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, bias=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(torch.relu(self.conv(x)))


class TinyTransformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.relu(self.proj(self.norm(x)))


class TestTorchMLIRImporter(unittest.TestCase):
    def test_import_cnn_preserves_parameter_names(self):
        model = TinyCNN()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.25)

        result = TorchMLIRImporter().import_module(
            model,
            torch.zeros(1, 1, 8, 8),
            func_name="tiny_cnn",
        )

        self.assertEqual(result.function_name, "tiny_cnn")
        self.assertIn("func.func @tiny_cnn", result.text)
        self.assertIn("torch.aten.convolution", result.text)
        self.assertIn("torch.aten.relu", result.text)
        self.assertEqual(
            set(result.parameter_map.values()),
            {"conv.weight", "conv.bias"},
        )
        self.assertEqual(result.buffer_map, {})
        self.assertTrue(result.mlir_module.operation.verify())

    def test_import_transformer_is_deterministic(self):
        model = TinyTransformerBlock(4)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.1)

        example = torch.zeros(2, 4)
        first = TorchMLIRImporter().import_module(
            model,
            example,
            func_name="transformer_block",
        )
        second = TorchMLIRImporter().import_module(
            model,
            example,
            func_name="transformer_block",
        )

        self.assertEqual(first.text, second.text)
        self.assertIn("torch.aten.var_mean", first.text)
        self.assertIn("torch.aten.mm", first.text)
        self.assertIn("torch.aten.relu", first.text)
        self.assertEqual(
            set(first.parameter_map.values()),
            {
                "norm.weight",
                "norm.bias",
                "proj.weight",
                "proj.bias",
            },
        )
        self.assertTrue(first.mlir_module.operation.verify())

    def test_frontend_uses_torch_mlir_path(self):
        model = TinyTransformerBlock(4)
        frontend = TorchFXFrontend(
            "fx_transformer",
            input_shapes={"x": ((2, 4), DType.F32)},
        )

        result = frontend.import_mlir(model)

        self.assertIn("func.func @fx_transformer", result.text)
        self.assertIn("torch.aten.relu", result.text)


if __name__ == "__main__":
    unittest.main()
