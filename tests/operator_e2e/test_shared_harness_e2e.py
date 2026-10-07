import unittest

import numpy as np

from compiler.frontends.manual_frontend import ManualFrontend
from compiler.model_ir import DType

from .helpers import (
    make_device,
    run_operator_e2e,
    vector_add_lowering,
)


def _vector_add_graph():
    frontend = ManualFrontend("vector_add")
    a = frontend.add_input("a", DType.F32, (8,))
    b = frontend.add_input("b", DType.F32, (8,))
    frontend.output(frontend.add(a, b))
    return frontend.build()


class TestSharedOperatorHarness(unittest.TestCase):
    """Prove that one reusable E2E pipeline drives the full compiler path."""

    def test_vector_add_uses_shared_harness(self):
        a = np.arange(8, dtype=np.float32) + 1.0
        b = np.arange(8, dtype=np.float32) * 2.0
        expected = [a + b]

        device = make_device()
        result = run_operator_e2e(
            device,
            _vector_add_graph(),
            {"a": a, "b": b},
            expected,
            lowering=vector_add_lowering,
            profile=True,
        )

        self.assertEqual(len(result.outputs), 1)
        np.testing.assert_allclose(result.outputs[0], expected[0])
        self.assertGreater(result.cycles, 0)
        self.assertIsNotNone(result.profile)
        self.assertIn("func.func @vector_add", result.mlir_text)
        self.assertIn("linalg.generic", result.mlir_text)
        self.assertIn("mgpu.launch", result.mlir_text)
        self.assertIn("LDG R1, [R0]", result.isa_text)
        self.assertIn("STG", result.isa_text)


if __name__ == "__main__":
    unittest.main()
