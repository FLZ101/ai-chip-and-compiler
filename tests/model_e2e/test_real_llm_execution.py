"""Execute a real pretrained GPT-2 component in the simulator.

This is a cropped static-shape milestone toward M5.  It proves that real
pretrained LLM normalization weights can run through the current C++ backend;
full greedy decode still requires multi-head attention at model scale.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from compiler.codegen import lower_kernel
from compiler.frontends import TorchFXFrontend
from compiler.model_ir import DType

from tests.operator_e2e.helpers import make_device, run_operator_e2e

GPT2_CACHE = (
    Path(__file__).resolve().parents[2] / ".cache" / "gpt2-small" / "pytorch_model.bin"
)


class GPT2LayerNormBlock(nn.Module):
    def __init__(self, layer_norm):
        super().__init__()
        self.layer_norm = layer_norm

    def forward(self, x):
        return self.layer_norm(x.reshape(8, 768)).reshape(1, 8, 768)


@unittest.skipUnless(
    GPT2_CACHE.exists() or os.environ.get("MGPU_DOWNLOAD_MODELS") == "1",
    "download GPT-2 small with scripts/download_real_models.py or set MGPU_DOWNLOAD_MODELS=1",
)
class TestRealLLMExecution(unittest.TestCase):
    def test_pretrained_gpt2_layer_norm_runs_in_simulator(self):
        from transformers import GPT2LMHeadModel

        model = GPT2LMHeadModel.from_pretrained(
            str(GPT2_CACHE.parent),
            local_files_only=True,
        ).eval()
        module = GPT2LayerNormBlock(model.transformer.h[0].ln_1).eval()
        graph_module = torch.fx.symbolic_trace(module)
        frontend = TorchFXFrontend(
            "gpt2_layer_norm",
            input_shapes={"x": ((1, 8, 768), DType.F32)},
        )
        graph = frontend.convert(graph_module)

        x = np.random.default_rng(0).normal(size=(1, 8, 768)).astype(np.float32)
        with torch.no_grad():
            expected = module(torch.from_numpy(x)).numpy()
        inputs = {
            "x": x,
            "layer_norm_weight": module.layer_norm.weight.detach().numpy(),
            "layer_norm_bias": module.layer_norm.bias.detach().numpy(),
        }

        result = run_operator_e2e(
            make_device(global_size=1 << 24, sm_count=16),
            graph,
            inputs,
            [expected],
            lowering=lower_kernel,
            atol=1e-4,
        )

        self.assertIn("mgpu.layer_norm", result.mlir_text)
        np.testing.assert_allclose(result.outputs[0], expected, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
