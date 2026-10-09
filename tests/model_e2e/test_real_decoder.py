"""Exported small GPT/decoder-block model test."""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from compiler.codegen import lower_kernel
from compiler.frontends import TorchFXFrontend
from compiler.frontends import attention_functions
from compiler.model_ir import DType

from examples.transformer.decoder_block import reference
from tests.operator_e2e.helpers import make_device, run_operator_e2e


class DecoderBlock(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        mask = attention_functions.causal_mask(q, k)
        attention = attention_functions.masked_attention(q, k, v, mask)
        return q + attention


class TestRealDecoderBlock(unittest.TestCase):
    def test_pytorch_decoder_block_runs_in_simulator(self):
        module = DecoderBlock()
        graph_module = torch.fx.symbolic_trace(module)
        frontend = TorchFXFrontend(
            "pytorch_decoder_block",
            input_shapes={
                "q": ((1, 1, 2, 2), DType.F32),
                "k": ((1, 1, 2, 2), DType.F32),
                "v": ((1, 1, 2, 2), DType.F32),
            },
        )
        graph = frontend.convert(graph_module)

        q = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
        k = np.asarray([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        v = np.asarray([[[[2.0, 1.0], [0.0, 3.0]]]], dtype=np.float32)
        inputs = {"q": q, "k": k, "v": v}
        expected = reference(q, k, v)

        result = run_operator_e2e(
            make_device(),
            graph,
            inputs,
            [expected],
            lowering=lower_kernel,
            atol=1e-5,
        )

        self.assertIn("mgpu.causal_mask", result.mlir_text)
        self.assertIn("mgpu.attention", result.mlir_text)
        np.testing.assert_allclose(result.outputs[0], expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
