import unittest

import numpy as np

from compiler.codegen import lower_common_kernel
from compiler.frontends.manual_frontend import ManualFrontend
from compiler.model_ir import DType

from .helpers import make_device, run_operator_e2e


def _elementwise_graph(op_name: str, n: int):
    frontend = ManualFrontend(f"{op_name}_{n}")
    a = frontend.add_input("a", DType.F32, (n,))
    b = frontend.add_input("b", DType.F32, (n,))
    if op_name == "add":
        output = frontend.add(a, b)
    elif op_name == "sub":
        output = frontend.sub(a, b)
    elif op_name == "mul":
        output = frontend.mul(a, b)
    elif op_name == "div":
        output = frontend.div(a, b)
    else:
        raise ValueError(op_name)
    frontend.output(output)
    return frontend.build()


def _reduction_graph(op_name: str, n: int):
    frontend = ManualFrontend(f"{op_name}_{n}")
    x = frontend.add_input("x", DType.F32, (n,))
    if op_name == "sum":
        output = frontend.reduce_sum(x)
    elif op_name == "mean":
        output = frontend.reduce_mean(x)
    elif op_name == "max":
        output = frontend.reduce_max(x)
    else:
        raise ValueError(op_name)
    frontend.output(output)
    return frontend.build()


def _matmul_graph(m: int, k: int, n: int):
    frontend = ManualFrontend("matmul")
    a = frontend.add_input("a", DType.F32, (m, k))
    b = frontend.add_input("b", DType.F32, (k, n))
    frontend.output(frontend.matmul(a, b))
    return frontend.build()


class TestCommonKernelsEndToEnd(unittest.TestCase):
    """Step 008 common-kernel tests through the shared E2E harness."""

    def _run_elementwise(self, op_name: str, n: int):
        a = np.arange(n, dtype=np.float32) + 1.0
        b = np.arange(n, dtype=np.float32) * 0.5 + 2.0
        if op_name == "add":
            expected = a + b
        elif op_name == "sub":
            expected = a - b
        elif op_name == "mul":
            expected = a * b
        elif op_name == "div":
            expected = a / b
        else:
            raise ValueError(op_name)

        result = run_operator_e2e(
            make_device(),
            _elementwise_graph(op_name, n),
            {"a": a, "b": b},
            [expected],
            lowering=lower_common_kernel,
        )
        self.assertIn("mgpu.launch", result.mlir_text)
        self.assertIn("ctaid.x", result.isa_text)
        np.testing.assert_allclose(result.outputs[0], expected, rtol=1e-5, atol=1e-6)

    def test_elementwise_ops(self):
        for op_name in ("add", "sub", "mul", "div"):
            with self.subTest(op=op_name):
                self._run_elementwise(op_name, 64)

    def test_reduction_ops(self):
        x = np.arange(64, dtype=np.float32) - 20.0
        cases = {
            "sum": np.sum(x),
            "mean": np.mean(x),
            "max": np.max(x),
        }
        for op_name, expected_value in cases.items():
            with self.subTest(op=op_name):
                result = run_operator_e2e(
                    make_device(),
                    _reduction_graph(op_name, x.shape[0]),
                    {"x": x},
                    [np.asarray(expected_value, dtype=np.float32)],
                    lowering=lower_common_kernel,
                )
                self.assertIn("BAR.SYNC", result.isa_text)
                np.testing.assert_allclose(
                    result.outputs[0],
                    np.asarray(expected_value, dtype=np.float32),
                    rtol=1e-5,
                    atol=1e-5,
                )

    def test_tiled_matmul(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(4, 8)).astype(np.float32)
        b = rng.normal(size=(8, 4)).astype(np.float32)
        expected = a @ b

        result = run_operator_e2e(
            make_device(),
            _matmul_graph(4, 8, 4),
            {"a": a, "b": b},
            [expected],
            lowering=lower_common_kernel,
        )

        self.assertIn("FMA", result.isa_text)
        self.assertIn("LDS", result.isa_text)
        self.assertIn("STS", result.isa_text)
        np.testing.assert_allclose(result.outputs[0], expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
