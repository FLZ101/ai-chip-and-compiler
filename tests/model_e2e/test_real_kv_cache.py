"""Exported KV-cache decode-step model test."""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from compiler.codegen import lower_kernel
from compiler.frontends import TorchFXFrontend
from compiler.frontends import cache_functions
from compiler.model_ir import DType

from tests.operator_e2e.helpers import make_device, run_operator_e2e


class KVDecodeStep(nn.Module):
    def forward(
        self,
        cache: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
    ):
        appended = cache_functions.kv_cache_append(cache, value)
        return cache_functions.kv_cache_gather(appended, positions)


class TestRealKVDecodeStep(unittest.TestCase):
    def test_pytorch_kv_decode_step_runs_in_simulator(self):
        module = KVDecodeStep()
        graph_module = torch.fx.symbolic_trace(module)
        frontend = TorchFXFrontend(
            "pytorch_kv_decode_step",
            input_shapes={
                "cache": ((1, 1, 4, 2), DType.F32),
                "value": ((1, 1, 4, 2), DType.F32),
                "positions": ((1, 1), DType.I64),
            },
        )
        graph = frontend.convert(graph_module)

        cache = np.zeros((1, 1, 4, 2), dtype=np.float32)
        value = np.arange(8, dtype=np.float32).reshape(1, 1, 4, 2)
        positions = np.asarray([[2]], dtype=np.int64)
        inputs = {"cache": cache, "value": value, "positions": positions}
        expected = value[:, :, 2:3, :]

        result = run_operator_e2e(
            make_device(),
            graph,
            inputs,
            [expected],
            lowering=lower_kernel,
            atol=1e-5,
        )

        self.assertIn("mgpu.kv_cache_append", result.mlir_text)
        self.assertIn("mgpu.kv_cache_gather", result.mlir_text)
        np.testing.assert_allclose(result.outputs[0], expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
