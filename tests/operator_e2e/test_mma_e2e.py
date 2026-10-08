import unittest

import numpy as np

from compiler.codegen import lower_common_kernel
from compiler.frontends.manual_frontend import ManualFrontend
from compiler.model_ir import DType
from hw.device import create_device, launch
from hw.isa import parse_asm

from .helpers import make_device, run_operator_e2e


def _matmul_graph():
    frontend = ManualFrontend("mma_matmul")
    a = frontend.add_input("a", DType.F32, (16, 16))
    b = frontend.add_input("b", DType.F32, (16, 16))
    frontend.output(frontend.matmul(a, b))
    return frontend.build()


class TestMMATileEndToEnd(unittest.TestCase):
    def test_16x16_mma_tile_matches_numpy(self):
        rng = np.random.default_rng(19)
        a = rng.normal(size=(16, 16)).astype(np.float32)
        b = rng.normal(size=(16, 16)).astype(np.float32)
        expected = a @ b

        result = run_operator_e2e(
            make_device(),
            _matmul_graph(),
            {"a": a, "b": b},
            [expected],
            lowering=lower_common_kernel,
        )

        self.assertIn("MMA", result.isa_text)
        np.testing.assert_allclose(result.outputs[0], expected, rtol=1e-5, atol=1e-5)

    def test_mma_rejects_partial_warp(self):
        device = create_device()
        program = parse_asm("""
            MOV R0, 0
            MOV R8, 0
            MOV R16, 0
            MMA R16, R0, R8, R16
            EXIT
            """)

        with self.assertRaises(NotImplementedError):
            launch(device, program, grid_dim=(1,), block_dim=(1,))


if __name__ == "__main__":
    unittest.main()
