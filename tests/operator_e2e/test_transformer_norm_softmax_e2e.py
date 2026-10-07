import unittest

import numpy as np

from compiler.codegen import lower_kernel
from compiler.frontends.manual_frontend import ManualFrontend
from compiler.model_ir import DType

from .helpers import make_device, run_operator_e2e


def _softmax_reference(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _layer_norm_reference(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def _rms_norm_reference(x, weight, eps=1e-5):
    rms = np.sqrt(np.mean(x * x, axis=1, keepdims=True) + eps)
    return x / rms * weight


class TestTransformerNormAndSoftmax(unittest.TestCase):
    """Step 012 Transformer normalization and softmax tests."""

    def test_softmax(self):
        x = np.array(
            [[1.0, 2.0, 3.0, 4.0], [1000.0, 1000.0, 1000.0, 1000.0]],
            dtype=np.float32,
        )
        frontend = ManualFrontend("softmax")
        value = frontend.add_input("x", DType.F32, x.shape)
        frontend.output(frontend.softmax(value, axis=-1))

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"x": x},
            [_softmax_reference(x)],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertIn("EXP2", result.isa_text)
        self.assertIn("MAX", result.isa_text)

    def test_layer_norm(self):
        x = np.arange(16, dtype=np.float32).reshape(4, 4)
        weight = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float32)
        bias = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)

        frontend = ManualFrontend("layer_norm")
        value = frontend.add_input("x", DType.F32, x.shape)
        w = frontend.add_input("w", DType.F32, weight.shape)
        b = frontend.add_input("b", DType.F32, bias.shape)
        frontend.output(frontend.layer_norm(value, w, b))

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"x": x, "w": weight, "b": bias},
            [_layer_norm_reference(x, weight, bias)],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertIn("RSQRT", result.isa_text)

    def test_rms_norm(self):
        x = np.arange(16, dtype=np.float32).reshape(4, 4)
        weight = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

        frontend = ManualFrontend("rms_norm")
        value = frontend.add_input("x", DType.F32, x.shape)
        w = frontend.add_input("w", DType.F32, weight.shape)
        frontend.output(frontend.rms_norm(value, w))

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"x": x, "w": weight},
            [_rms_norm_reference(x, weight)],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertIn("RSQRT", result.isa_text)


if __name__ == "__main__":
    unittest.main()
