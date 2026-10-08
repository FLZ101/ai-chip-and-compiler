import unittest

import numpy as np

from compiler.frontends.manual_frontend import ManualFrontend
from compiler.mlir import MLIRBuilder, run_mgpu_pipeline
from compiler.model_ir import DType
from hw.device import create_device, launch
from hw.isa import parse_asm


def _vector_add_graph():
    frontend = ManualFrontend("vector_add")
    a = frontend.add_input("a", DType.F32, (8,))
    b = frontend.add_input("b", DType.F32, (8,))
    frontend.output(frontend.add(a, b))
    return frontend.build()


class TestMgpuOptPipelineEndToEnd(unittest.TestCase):
    """Phase 0: run the C++ mgpu-to-isa backend for vector add."""

    def test_vector_add_uses_mgpu_opt_isa(self):
        graph = _vector_add_graph()
        mlir_text = MLIRBuilder().build(graph)
        result = run_mgpu_pipeline(mlir_text)

        a = np.arange(8, dtype=np.float32) + 1.0
        b = np.arange(8, dtype=np.float32) * 2.0
        expected = a + b

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, a.tolist())
        device.global_memory.load_floats(32, b.tolist())
        device.global_memory.load_floats(64, [0.0] * len(a))

        program = parse_asm(result.code)
        launch(device, program, grid_dim=result.grid, block_dim=result.block)

        actual = np.asarray(
            device.global_memory.read_floats(64, len(a)), dtype=np.float32
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(result.grid, (1, 1, 1))
        self.assertEqual(result.block, (32, 1, 1))


if __name__ == "__main__":
    unittest.main()
