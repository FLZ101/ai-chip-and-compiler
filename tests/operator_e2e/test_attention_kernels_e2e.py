import unittest

import numpy as np

from compiler.codegen import lower_kernel
from compiler.frontends.manual_frontend import ManualFrontend
from compiler.model_ir import DType

from .helpers import make_device, run_operator_e2e


def _softmax_reference(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _attention_reference(q, k, v, scale, mask=None):
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    if mask is not None:
        scores = np.where(mask == 0, -np.inf, scores)
    weights = _softmax_reference(scores)
    return np.matmul(weights, v)


def _rope_reference(x, cos, sin):
    output = np.zeros_like(x)
    for i in range(0, x.shape[-1], 2):
        x0 = x[..., i]
        x1 = x[..., i + 1]
        c = cos[..., i]
        s = sin[..., i]
        output[..., i] = x0 * c - x1 * s
        output[..., i + 1] = x0 * s + x1 * c
    return output


class TestAttentionKernels(unittest.TestCase):
    """Step 013 attention, RoPE, and KV-cache tests."""

    def test_causal_mask(self):
        frontend = ManualFrontend("causal_mask")
        frontend.output(frontend.causal_mask(query_len=3, key_len=3, batch=1))

        expected = np.tril(np.ones((3, 3), dtype=np.float32)).reshape(1, 1, 3, 3)
        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {},
            [expected],
            lowering=lower_kernel,
        )
        np.testing.assert_array_equal(result.outputs[0], expected)

    def test_rope(self):
        x = np.arange(8, dtype=np.float32).reshape(2, 4)
        cos = np.array([[0.9, 0.9, 0.8, 0.8], [0.7, 0.7, 0.6, 0.6]], dtype=np.float32)
        sin = np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]], dtype=np.float32)
        frontend = ManualFrontend("rope")
        value = frontend.add_input("x", DType.F32, x.shape)
        c = frontend.add_input("cos", DType.F32, cos.shape)
        s = frontend.add_input("sin", DType.F32, sin.shape)
        frontend.output(frontend.rope(value, c, s))

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"x": x, "cos": cos, "sin": sin},
            [_rope_reference(x, cos, sin)],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertIn("SEL", result.isa_text)

    def test_scaled_dot_product_attention(self):
        q = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
        k = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        v = np.array([[[[2.0, 1.0], [0.0, 3.0]]]], dtype=np.float32)
        scale = 1.0 / np.sqrt(2.0)
        frontend = ManualFrontend("attention")
        qv = frontend.add_input("q", DType.F32, q.shape)
        kv = frontend.add_input("k", DType.F32, k.shape)
        vv = frontend.add_input("v", DType.F32, v.shape)
        frontend.output(frontend.scaled_dot_product_attention(qv, kv, vv, scale=scale))

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"q": q, "k": k, "v": v},
            [_attention_reference(q, k, v, scale)],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertIn("EXP2", result.isa_text)
        self.assertIn("MMA", result.isa_text)

    def test_masked_attention(self):
        q = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
        k = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        v = np.array([[[[2.0, 1.0], [0.0, 3.0]]]], dtype=np.float32)
        mask = np.array([[[[1.0, 0.0], [1.0, 1.0]]]], dtype=np.float32)
        scale = 1.0 / np.sqrt(2.0)
        frontend = ManualFrontend("masked_attention")
        qv = frontend.add_input("q", DType.F32, q.shape)
        kv = frontend.add_input("k", DType.F32, k.shape)
        vv = frontend.add_input("v", DType.F32, v.shape)
        mv = frontend.add_input("mask", DType.F32, mask.shape)
        frontend.output(frontend.masked_attention(qv, kv, vv, mv, scale=scale))

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"q": q, "k": k, "v": v, "mask": mask},
            [_attention_reference(q, k, v, scale, mask)],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertIn("EXP2", result.isa_text)
        self.assertIn("MMA", result.isa_text)

    def test_kv_cache_append(self):
        cache = np.zeros((1, 1, 2, 2), dtype=np.float32)
        value = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
        frontend = ManualFrontend("kv_cache_append")
        c = frontend.add_input("cache", DType.F32, cache.shape)
        v = frontend.add_input("value", DType.F32, value.shape)
        frontend.output(frontend.kv_cache_append(c, v))

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"cache": cache, "value": value},
            [value],
            lowering=lower_kernel,
        )
        np.testing.assert_array_equal(result.outputs[0], value)

    def test_kv_cache_gather(self):
        cache = np.arange(8, dtype=np.float32).reshape(1, 1, 4, 2)
        positions = np.array([[2]], dtype=np.float32)
        frontend = ManualFrontend("kv_cache_gather")
        c = frontend.add_input("cache", DType.F32, cache.shape)
        p = frontend.add_input("positions", DType.I64, positions.shape)
        frontend.output(frontend.kv_cache_gather(c, p))

        expected = cache[:, :, 2:3, :]
        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"cache": cache, "positions": positions},
            [expected],
            lowering=lower_kernel,
        )
        np.testing.assert_array_equal(result.outputs[0], expected)


if __name__ == "__main__":
    unittest.main()
