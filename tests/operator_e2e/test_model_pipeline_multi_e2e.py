"""Multi-launch model-pipeline tests for ``run_model_pipeline``."""

from __future__ import annotations

import unittest

import numpy as np

from compiler.mlir import (
    MGpuIsaBundle,
    MGpuIsaResult,
    MGpuOptError,
    enqueue_isa_bundle,
    run_mgpu_pipeline,
    run_model_pipeline,
)
from runtime.driver import Driver

_TWO_MATMUL_MODULE = """
module {
  func.func @first(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %init = tensor.empty() : tensor<4x4xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%fill : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
  func.func @second(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %init = tensor.empty() : tensor<4x4xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%fill : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
"""

_UNSUPPORTED_LAUNCH_MODULE = """
module {
  func.func @unsupported(%arg0: memref<4xf32>) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c32, %c1, %c1) {
      %cst = arith.constant 42 : i32
      mgpu.return
    }
    return
  }
}
"""


class TestModelPipelineMultiLaunch(unittest.TestCase):
    def test_full_module_pipeline_extracts_every_isa_op(self):
        bundle = run_model_pipeline(_TWO_MATMUL_MODULE)
        self.assertIsInstance(bundle, MGpuIsaBundle)
        self.assertEqual(len(bundle.programs), 2)
        self.assertTrue(all(program.code for program in bundle.programs))
        self.assertTrue(all(program.grid for program in bundle.programs))
        self.assertTrue(all(program.block for program in bundle.programs))

    def test_enqueue_isa_bundle_runs_programs_in_dependency_order(self):
        driver = Driver(global_size=1 << 16, sm_count=2)
        bundle = MGpuIsaBundle(
            programs=(
                MGpuIsaResult(
                    code=("MOV R0, 3\n" "MOV R1, 0\n" "STG [R1], R0\n" "EXIT\n"),
                    grid=(1,),
                    block=(1,),
                ),
                MGpuIsaResult(
                    code=(
                        "MOV R2, 0\n"
                        "LDG R1, [R2]\n"
                        "ADD R1, R1, 2\n"
                        "MOV R2, 4\n"
                        "STG [R2], R1\n"
                        "EXIT\n"
                    ),
                    grid=(1,),
                    block=(1,),
                ),
            )
        )

        enqueue_isa_bundle(driver, bundle)
        driver.synchronize()

        first = driver.device.global_memory.read_floats(0, 1)[0]
        second = driver.device.global_memory.read_floats(4, 1)[0]
        np.testing.assert_allclose([first, second], [3.0, 5.0])

    def test_unsupported_launch_reports_op_diagnostic(self):
        with self.assertRaisesRegex(MGpuOptError, "cannot lower mgpu.launch"):
            run_mgpu_pipeline(
                _UNSUPPORTED_LAUNCH_MODULE,
                pipeline="builtin.module(func.func(mgpu-to-isa))",
            )


if __name__ == "__main__":
    unittest.main()
