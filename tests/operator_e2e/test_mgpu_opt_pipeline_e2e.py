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

    def _run_elementwise_via_mgpu_opt(self, op_name: str, n: int):
        frontend = ManualFrontend(op_name)
        a = frontend.add_input("a", DType.F32, (n,))
        b = frontend.add_input("b", DType.F32, (n,))
        if op_name == "sub":
            output = frontend.sub(a, b)
        elif op_name == "mul":
            output = frontend.mul(a, b)
        elif op_name == "div":
            output = frontend.div(a, b)
        else:
            raise ValueError(op_name)
        frontend.output(output)
        graph = frontend.build()

        result = run_mgpu_pipeline(MLIRBuilder().build(graph))
        a_data = np.arange(n, dtype=np.float32) + 1.0
        b_data = np.arange(n, dtype=np.float32) * 0.5 + 2.0
        if op_name == "sub":
            expected = a_data - b_data
        elif op_name == "mul":
            expected = a_data * b_data
        else:
            expected = a_data / b_data

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, a_data.tolist())
        device.global_memory.load_floats(n * 4, b_data.tolist())
        device.global_memory.load_floats(2 * n * 4, [0.0] * n)

        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(2 * n * 4, n), dtype=np.float32
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_elementwise_ops_use_mgpu_opt(self):
        for op_name in ("sub", "mul", "div"):
            with self.subTest(op=op_name):
                self._run_elementwise_via_mgpu_opt(op_name, 8)

    def _run_reduction_via_mgpu_opt(self, op_name: str, n: int):
        frontend = ManualFrontend(op_name)
        x = frontend.add_input("x", DType.F32, (n,))
        if op_name == "sum":
            output = frontend.reduce_sum(x)
        elif op_name == "mean":
            output = frontend.reduce_mean(x)
        else:
            output = frontend.reduce_max(x)
        frontend.output(output)
        graph = frontend.build()

        result = run_mgpu_pipeline(MLIRBuilder().build(graph))
        data = np.arange(n, dtype=np.float32) - 3.0
        if op_name == "sum":
            expected = np.sum(data)
        elif op_name == "mean":
            expected = np.mean(data)
        else:
            expected = np.max(data)

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, data.tolist())
        device.global_memory.load_floats(n * 4, [0.0])

        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = device.global_memory.read_floats(n * 4, 1)[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_reductions_use_mgpu_opt(self):
        for op_name in ("sum", "mean", "max"):
            with self.subTest(op=op_name):
                self._run_reduction_via_mgpu_opt(op_name, 8)


if __name__ == "__main__":
    unittest.main()
