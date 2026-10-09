"""Exported two-step autoregressive decode-loop model test."""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from compiler.codegen import lower_kernel
from compiler.frontends import TorchFXFrontend
from compiler.frontends import cache_functions
from compiler.model_ir import DType

from examples.transformer.autoregressive_loop import reference
from tests.operator_e2e.helpers import make_device, run_operator_e2e


class AutoregressiveLoop(nn.Module):
    def forward(
        self,
        cache: torch.Tensor,
        pos0: torch.Tensor,
        pos1: torch.Tensor,
    ):
        first = cache_functions.kv_cache_gather(cache, pos0)
        second = cache_functions.kv_cache_gather(cache, pos1)
        return first, second


class TestRealAutoregressiveLoop(unittest.TestCase):
    def test_pytorch_autoregressive_loop_runs_in_simulator(self):
        module = AutoregressiveLoop()
        graph_module = torch.fx.symbolic_trace(module)
        frontend = TorchFXFrontend(
            "pytorch_autoregressive_loop",
            input_shapes={
                "cache": ((1, 1, 4, 2), DType.F32),
                "pos0": ((1, 1), DType.I64),
                "pos1": ((1, 1), DType.I64),
            },
        )
        graph = frontend.convert(graph_module)

        cache = np.arange(8, dtype=np.float32).reshape(1, 1, 4, 2)
        pos0 = np.asarray([[1]], dtype=np.int64)
        pos1 = np.asarray([[3]], dtype=np.int64)
        inputs = {"cache": cache, "pos0": pos0, "pos1": pos1}
        expected = reference(cache, pos0, pos1)

        result = run_operator_e2e(
            make_device(),
            graph,
            inputs,
            expected,
            lowering=lower_kernel,
            atol=1e-5,
        )

        self.assertIn("mgpu.kv_cache_gather", result.mlir_text)
        self.assertEqual(len(result.outputs), 2)
        np.testing.assert_allclose(result.outputs[0], expected[0], atol=1e-5)
        np.testing.assert_allclose(result.outputs[1], expected[1], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
