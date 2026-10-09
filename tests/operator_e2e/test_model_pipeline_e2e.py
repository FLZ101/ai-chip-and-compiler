"""M4.5 linalg-on-tensors model pipeline test."""

from __future__ import annotations

import unittest

import numpy as np

from compiler.mlir import run_model_pipeline
from hw.device import create_device, launch
from hw.isa import parse_asm

_MATMUL_MODULE = """
module {
  func.func @matmul(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %init = tensor.empty() : tensor<16x16xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%fill : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }
}
"""


class TestModelPipelineEndToEnd(unittest.TestCase):
    def test_linalg_matmul_lowers_and_runs(self):
        result = run_model_pipeline(_MATMUL_MODULE)
        self.assertIn("MMA", result.code)

        rng = np.random.default_rng(5)
        a = rng.normal(size=(16, 16)).astype(np.float32)
        b = rng.normal(size=(16, 16)).astype(np.float32)
        expected = a @ b

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(2048, a.reshape(-1).tolist())
        device.global_memory.load_floats(1024, b.reshape(-1).tolist())

        launch(
            device,
            parse_asm(result.code),
            grid_dim=result.grid,
            block_dim=result.block,
        )

        actual = np.asarray(
            device.global_memory.read_floats(0, 256), dtype=np.float32
        ).reshape(16, 16)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
