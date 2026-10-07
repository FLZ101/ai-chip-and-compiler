import unittest

import numpy as np

from compiler.codegen import lower_kernel
from compiler.frontends.manual_frontend import ManualFrontend
from compiler.model_ir import DType

from .helpers import make_device, run_operator_e2e


def _activation_graph(op_name: str, n: int):
    frontend = ManualFrontend(f"{op_name}_{n}")
    x = frontend.add_input("x", DType.F32, (n,))
    if op_name == "relu":
        output = frontend.relu(x)
    elif op_name == "leaky_relu":
        output = frontend.leaky_relu(x, alpha=0.01)
    elif op_name == "silu":
        output = frontend.silu(x)
    elif op_name == "gelu":
        output = frontend.gelu(x)
    elif op_name == "sigmoid":
        output = frontend.sigmoid(x)
    else:
        raise ValueError(op_name)
    frontend.output(output)
    return frontend.build()


class TestActivationAndShapeKernels(unittest.TestCase):
    """Step 009 activation and shape/layout operator tests."""

    def test_activations(self):
        x = np.linspace(-4.0, 4.0, 64, dtype=np.float32)

        def sigmoid(v):
            return 1.0 / (1.0 + np.exp(-v))

        gelu_approx = (
            0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
        )
        cases = {
            "relu": np.maximum(x, 0.0),
            "leaky_relu": np.where(x < 0, 0.01 * x, x),
            "silu": x * sigmoid(x),
            "gelu": gelu_approx,
            "sigmoid": sigmoid(x),
        }

        for op_name, expected in cases.items():
            with self.subTest(op=op_name):
                result = run_operator_e2e(
                    make_device(),
                    _activation_graph(op_name, x.shape[0]),
                    {"x": x},
                    [np.asarray(expected, dtype=np.float32)],
                    lowering=lower_kernel,
                    rtol=1e-5,
                    atol=1e-5,
                )
                self.assertIn("linalg.generic", result.mlir_text)
                np.testing.assert_allclose(
                    result.outputs[0],
                    np.asarray(expected, dtype=np.float32),
                    rtol=1e-5,
                    atol=1e-5,
                )

    def test_reshape(self):
        x = np.arange(16, dtype=np.float32)
        frontend = ManualFrontend("reshape")
        value = frontend.add_input("x", DType.F32, (16,))
        frontend.output(frontend.reshape(value, (4, 4)))

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"x": x},
            [x.reshape(4, 4)],
            lowering=lower_kernel,
        )
        np.testing.assert_array_equal(result.outputs[0], x.reshape(4, 4))

    def test_transpose(self):
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        frontend = ManualFrontend("transpose")
        value = frontend.add_input("x", DType.F32, (2, 3))
        frontend.output(frontend.transpose(value, (1, 0)))

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"x": x},
            [x.T],
            lowering=lower_kernel,
        )
        np.testing.assert_array_equal(result.outputs[0], x.T)

    def test_concat(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        frontend = ManualFrontend("concat")
        left = frontend.add_input("a", DType.F32, (3,))
        right = frontend.add_input("b", DType.F32, (4,))
        frontend.output(frontend.concat(left, right, axis=0))

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"a": a, "b": b},
            [np.concatenate([a, b])],
            lowering=lower_kernel,
        )
        np.testing.assert_array_equal(result.outputs[0], np.concatenate([a, b]))

    def test_slice(self):
        x = np.arange(8, dtype=np.float32)
        frontend = ManualFrontend("slice")
        value = frontend.add_input("x", DType.F32, (8,))
        frontend.output(frontend.slice(value, (2,), (6,), steps=(1,)))

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"x": x},
            [x[2:6]],
            lowering=lower_kernel,
        )
        np.testing.assert_array_equal(result.outputs[0], x[2:6])

    def test_pad(self):
        x = np.arange(4, dtype=np.float32)
        frontend = ManualFrontend("pad")
        value = frontend.add_input("x", DType.F32, (4,))
        frontend.output(frontend.pad(value, ((1, 2),), value=-1.0))

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"x": x},
            [np.pad(x, (1, 2), constant_values=-1.0)],
            lowering=lower_kernel,
        )
        np.testing.assert_array_equal(
            result.outputs[0], np.pad(x, (1, 2), constant_values=-1.0)
        )

    def test_resize(self):
        x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        frontend = ManualFrontend("resize")
        value = frontend.add_input("x", DType.F32, (4,))
        frontend.output(frontend.resize(value, sizes=(8,)))

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"x": x},
            [np.repeat(x, 2)],
            lowering=lower_kernel,
        )
        np.testing.assert_array_equal(result.outputs[0], np.repeat(x, 2))


if __name__ == "__main__":
    unittest.main()
