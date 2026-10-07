import unittest

import numpy as np

from compiler.frontends.manual_frontend import ManualFrontend
from compiler.mlir import lower_mgpu_to_isa
from compiler.model_ir import DType

from tests.operator_e2e.helpers import make_device, run_operator_e2e


def _vector_add_graph():
    frontend = ManualFrontend("vector_add")
    a = frontend.add_input("a", DType.F32, (8,))
    b = frontend.add_input("b", DType.F32, (8,))
    frontend.output(frontend.add(a, b))
    return frontend.build()


def _matmul_graph(m, k, n):
    frontend = ManualFrontend("matmul")
    a = frontend.add_input("a", DType.F32, (m, k))
    b = frontend.add_input("b", DType.F32, (k, n))
    frontend.output(frontend.matmul(a, b))
    return frontend.build()


class TestMGpuToIsa(unittest.TestCase):
    def test_vector_add_runs_in_simulator(self):
        graph = _vector_add_graph()
        a = np.arange(8, dtype=np.float32)
        b = np.ones(8, dtype=np.float32)
        result = run_operator_e2e(
            make_device(),
            graph,
            {"a": a, "b": b},
            [a + b],
            lowering=lower_mgpu_to_isa,
        )
        self.assertIn("mgpu.launch", result.mlir_text)
        self.assertIn("LDG", result.isa_text)
        self.assertIn("STG", result.isa_text)
        np.testing.assert_allclose(result.outputs[0], a + b, atol=1e-5)

    def test_matmul_runs_in_simulator(self):
        rng = np.random.default_rng(7)
        a = rng.normal(size=(4, 8)).astype(np.float32)
        b = rng.normal(size=(8, 4)).astype(np.float32)
        graph = _matmul_graph(4, 8, 4)
        result = run_operator_e2e(
            make_device(),
            graph,
            {"a": a, "b": b},
            [a @ b],
            lowering=lower_mgpu_to_isa,
        )
        self.assertIn("mgpu.launch", result.mlir_text)
        self.assertIn("FMA", result.isa_text)
        self.assertIn("LDS", result.isa_text)
        np.testing.assert_allclose(result.outputs[0], a @ b, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
