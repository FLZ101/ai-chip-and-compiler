"""Differential tests for native software operator kernels."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

from compiler.codegen import lower_kernel
from compiler.frontends.manual_frontend import ManualFrontend
from compiler.model_ir import DType

from .helpers import make_device, run_operator_e2e, run_software_model_e2e

CPP_SIM_DIR = Path(__file__).resolve().parents[2] / ".deps" / "cpp-sim-build"
if str(CPP_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(CPP_SIM_DIR))

import cpp_sim  # noqa: E402


def _conv2d_reference(image, weight, bias, *, stride=1, padding=0, dilation=1):
    n, c, h, w = image.shape
    f, _, kh, kw = weight.shape
    effective_kh = dilation * (kh - 1) + 1
    effective_kw = dilation * (kw - 1) + 1
    out_h = (h + 2 * padding - effective_kh) // stride + 1
    out_w = (w + 2 * padding - effective_kw) // stride + 1
    output = np.zeros((n, f, out_h, out_w), dtype=np.float32)
    for ff in range(f):
        for oy in range(out_h):
            for ox in range(out_w):
                acc = bias[ff]
                for cc in range(c):
                    for ky in range(kh):
                        iy = oy * stride + ky * dilation - padding
                        if not 0 <= iy < h:
                            continue
                        for kx in range(kw):
                            ix = ox * stride + kx * dilation - padding
                            if 0 <= ix < w:
                                acc += image[0, cc, iy, ix] * weight[ff, cc, ky, kx]
                output[0, ff, oy, ox] = acc
    return output


class TestSoftwareKernelsEndToEnd(unittest.TestCase):
    def _assert_equivalent(self, graph, inputs, expected, *, rtol=1e-5, atol=1e-5):
        software = run_software_model_e2e(
            make_device(global_size=1 << 24, sm_count=4),
            graph,
            inputs,
            [expected],
            rtol=rtol,
            atol=atol,
        )
        mgpu = run_operator_e2e(
            make_device(global_size=1 << 24, sm_count=4),
            graph,
            inputs,
            [expected],
            lowering=lower_kernel,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            software.outputs[0],
            mgpu.outputs[0],
            rtol=rtol,
            atol=atol,
        )

    def test_conv2d_software_matches_mgpu_and_reference(self):
        rng = np.random.default_rng(12)
        image = rng.normal(size=(1, 2, 6, 6)).astype(np.float32)
        weight = rng.normal(size=(2, 2, 3, 3)).astype(np.float32)
        bias = rng.normal(size=(2,)).astype(np.float32)
        stride = 1
        padding = 0
        dilation = 1

        frontend = ManualFrontend("conv2d")
        x = frontend.add_input("x", DType.F32, image.shape)
        w = frontend.add_input("w", DType.F32, weight.shape)
        b = frontend.add_input("b", DType.F32, bias.shape)
        frontend.output(
            frontend.conv2d(
                x,
                w,
                b,
                strides=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        graph = frontend.build()
        output_shape = graph.outputs[0].shape
        expected = _conv2d_reference(
            image,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        device = make_device(global_size=1 << 20, sm_count=4)
        layout = {
            "x": (0, image.shape),
            "w": (image.nbytes, weight.shape),
            "b": (image.nbytes + weight.nbytes, bias.shape),
            graph.outputs[0].name: (
                image.nbytes + weight.nbytes + bias.nbytes,
                output_shape,
            ),
        }
        for name, (offset, shape) in layout.items():
            array = {
                "x": image,
                "w": weight,
                "b": bias,
                graph.outputs[0].name: np.zeros(shape, dtype=np.float32),
            }[name]
            device.global_memory.load_floats(offset, array.reshape(-1).tolist())

        request = {
            "op": "conv2d",
            "input_shape": image.shape,
            "weight_shape": weight.shape,
            "output_shape": output_shape,
            "strides": (stride, stride),
            "paddings": (padding, padding),
            "dilations": (dilation, dilation),
            "has_bias": True,
            "input_offset": layout["x"][0],
            "weight_offset": layout["w"][0],
            "bias_offset": layout["b"][0],
            "output_offset": layout[graph.outputs[0].name][0],
        }
        self.assertTrue(cpp_sim.software_run(device.global_memory.data, request))
        software_output = np.asarray(
            device.global_memory.read_floats(
                layout[graph.outputs[0].name][0],
                np.prod(output_shape),
            ),
            dtype=np.float32,
        ).reshape(output_shape)

        np.testing.assert_allclose(software_output, expected, rtol=1e-5, atol=1e-5)

        mgpu_result = run_operator_e2e(
            make_device(global_size=1 << 20, sm_count=4),
            graph,
            {"x": image, "w": weight, "b": bias},
            [expected],
            lowering=__import__(
                "compiler.codegen", fromlist=["lower_kernel"]
            ).lower_kernel,
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            software_output,
            mgpu_result.outputs[0],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_matmul_software_matches_mgpu(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(2, 3)).astype(np.float32)
        b = rng.normal(size=(3, 4)).astype(np.float32)
        frontend = ManualFrontend("matmul")
        lhs = frontend.add_input("a", DType.F32, a.shape)
        rhs = frontend.add_input("b", DType.F32, b.shape)
        frontend.output(frontend.matmul(lhs, rhs))
        self._assert_equivalent(frontend.build(), {"a": a, "b": b}, a @ b)

    def test_linear_software_matches_mgpu(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(2, 3)).astype(np.float32)
        w = rng.normal(size=(4, 3)).astype(np.float32)
        bias = rng.normal(size=(4,)).astype(np.float32)
        frontend = ManualFrontend("linear")
        value = frontend.add_input("x", DType.F32, x.shape)
        weight = frontend.add_input("w", DType.F32, w.shape)
        b = frontend.add_input("b", DType.F32, bias.shape)
        frontend.output(frontend.linear(value, weight, b))
        self._assert_equivalent(
            frontend.build(),
            {"x": x, "w": w, "b": bias},
            x @ w.T + bias,
        )

    def test_relu_software_matches_mgpu(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(4, 6)).astype(np.float32)
        frontend = ManualFrontend("relu")
        value = frontend.add_input("x", DType.F32, x.shape)
        frontend.output(frontend.relu(value))
        self._assert_equivalent(
            frontend.build(),
            {"x": x},
            np.maximum(x, 0).astype(np.float32),
        )

    def test_add_software_matches_mgpu(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=(4, 5)).astype(np.float32)
        b = rng.normal(size=(4, 5)).astype(np.float32)
        frontend = ManualFrontend("add")
        lhs = frontend.add_input("a", DType.F32, a.shape)
        rhs = frontend.add_input("b", DType.F32, b.shape)
        frontend.output(frontend.add(lhs, rhs))
        self._assert_equivalent(frontend.build(), {"a": a, "b": b}, a + b)

    def test_positional_broadcast_add_software_reference(self):
        token = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        positional = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        frontend = ManualFrontend("positional_add")
        x = frontend.add_input("token", DType.F32, token.shape)
        pos = frontend.add_input("pos", DType.F32, positional.shape)
        frontend.output(frontend.add(x, pos))
        expected = token + positional
        result = run_software_model_e2e(
            make_device(global_size=1 << 16, sm_count=4),
            frontend.build(),
            {"token": token, "pos": positional},
            [expected],
        )
        np.testing.assert_allclose(result.outputs[0], expected)

    def test_seq_hidden_broadcast_add_software_reference(self):
        token = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        positional = np.arange(12, dtype=np.float32).reshape(3, 4)
        frontend = ManualFrontend("seq_positional_add")
        x = frontend.add_input("token", DType.F32, token.shape)
        pos = frontend.add_input("pos", DType.F32, positional.shape)
        frontend.output(frontend.add(x, pos))
        expected = token + positional
        result = run_software_model_e2e(
            make_device(global_size=1 << 16, sm_count=4),
            frontend.build(),
            {"token": token, "pos": positional},
            [expected],
        )
        np.testing.assert_allclose(result.outputs[0], expected)

    def test_batch_norm_software_matches_mgpu(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(1, 2, 3, 3)).astype(np.float32)
        scale = rng.normal(size=(2,)).astype(np.float32) + 1.0
        bias = rng.normal(size=(2,)).astype(np.float32)
        mean = rng.normal(size=(2,)).astype(np.float32)
        var = rng.uniform(0.5, 2.0, size=(2,)).astype(np.float32)
        eps = 1e-5
        frontend = ManualFrontend("batch_norm")
        value = frontend.add_input("x", DType.F32, x.shape)
        s = frontend.add_input("s", DType.F32, scale.shape)
        b = frontend.add_input("b", DType.F32, bias.shape)
        m = frontend.add_input("m", DType.F32, mean.shape)
        v = frontend.add_input("v", DType.F32, var.shape)
        frontend.output(frontend.batch_norm(value, s, b, m, v, epsilon=eps))
        expected = (
            (x - mean.reshape(1, 2, 1, 1)) / np.sqrt(var.reshape(1, 2, 1, 1) + eps)
        ) * scale.reshape(1, 2, 1, 1) + bias.reshape(1, 2, 1, 1)
        self._assert_equivalent(
            frontend.build(),
            {"x": x, "s": scale, "b": bias, "m": mean, "v": var},
            expected,
        )

    def test_pooling_software_matches_mgpu(self):
        image = np.arange(1 * 2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4)
        for op_name in ("max_pool2d", "avg_pool2d"):
            with self.subTest(op=op_name):
                frontend = ManualFrontend(op_name)
                x = frontend.add_input("x", DType.F32, image.shape)
                if op_name == "max_pool2d":
                    frontend.output(frontend.max_pool2d(x, kernel=2, strides=2))
                    expected = image.reshape(1, 2, 2, 2, 2, 2).max(axis=(3, 5))
                else:
                    frontend.output(frontend.avg_pool2d(x, kernel=2, strides=2))
                    expected = image.reshape(1, 2, 2, 2, 2, 2).mean(axis=(3, 5))
                self._assert_equivalent(frontend.build(), {"x": image}, expected)

    def test_global_avg_pool_software_matches_mgpu(self):
        image = np.arange(1 * 2 * 3 * 3, dtype=np.float32).reshape(1, 2, 3, 3)
        frontend = ManualFrontend("global_avg_pool")
        x = frontend.add_input("x", DType.F32, image.shape)
        frontend.output(frontend.global_avg_pool(x))
        self._assert_equivalent(
            frontend.build(),
            {"x": image},
            image.mean(axis=(2, 3), keepdims=True),
        )

    def test_pad_software_matches_mgpu(self):
        image = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)
        frontend = ManualFrontend("pad")
        x = frontend.add_input("x", DType.F32, image.shape)
        frontend.output(
            frontend.pad(
                x,
                ((0, 0), (0, 0), (1, 1), (1, 1)),
                value=-1.0,
            )
        )
        expected = np.full((1, 2, 4, 4), -1.0, dtype=np.float32)
        expected[:, :, 1:3, 1:3] = image
        self._assert_equivalent(frontend.build(), {"x": image}, expected)

    def test_reshape_software_matches_mgpu(self):
        image = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)
        frontend = ManualFrontend("reshape")
        x = frontend.add_input("x", DType.F32, image.shape)
        frontend.output(frontend.reshape(x, (1, 8)))
        self._assert_equivalent(frontend.build(), {"x": image}, image.reshape(1, 8))

    def test_embedding_gather_software_reference(self):
        indices = np.asarray([[0, 2], [1, 3]], dtype=np.int32)
        weight = np.arange(20, dtype=np.float32).reshape(5, 4)
        frontend = ManualFrontend("embedding_gather")
        ids = frontend.add_input("indices", DType.I32, indices.shape)
        w = frontend.add_input("weight", DType.F32, weight.shape)
        frontend.output(frontend.embedding_gather(ids, w))
        result = run_software_model_e2e(
            make_device(global_size=1 << 16, sm_count=4),
            frontend.build(),
            {"indices": indices, "weight": weight},
            [weight[indices]],
        )
        np.testing.assert_array_equal(result.outputs[0], weight[indices])

    def test_argmax_software_reference(self):
        logits = np.asarray([[0.2, 1.3, 0.7], [2.1, 0.4, 0.9]], dtype=np.float32)
        frontend = ManualFrontend("argmax")
        x = frontend.add_input("x", DType.F32, logits.shape)
        frontend.output(frontend.argmax(x, axis=-1))
        result = run_software_model_e2e(
            make_device(global_size=1 << 16, sm_count=4),
            frontend.build(),
            {"x": logits},
            [np.argmax(logits, axis=-1).astype(np.int32)],
        )
        np.testing.assert_array_equal(
            result.outputs[0], np.argmax(logits, axis=-1).astype(np.int32)
        )

    def test_large_attention_software_reference(self):
        rng = np.random.default_rng(17)
        q = rng.normal(size=(1, 12, 8, 64)).astype(np.float32)
        k = rng.normal(size=(1, 12, 8, 64)).astype(np.float32)
        v = rng.normal(size=(1, 12, 8, 64)).astype(np.float32)
        mask = np.tril(np.ones((8, 8), dtype=np.float32)).reshape(1, 1, 8, 8)
        scale = 1.0 / np.sqrt(64.0)
        frontend = ManualFrontend("masked_attention_large")
        qv = frontend.add_input("q", DType.F32, q.shape)
        kv = frontend.add_input("k", DType.F32, k.shape)
        vv = frontend.add_input("v", DType.F32, v.shape)
        mv = frontend.add_input("mask", DType.F32, mask.shape)
        frontend.output(frontend.masked_attention(qv, kv, vv, mv, scale=scale))

        scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
        scores = np.where(mask == 0, -np.inf, scores)
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        expected = np.matmul(weights, v)

        result = run_software_model_e2e(
            make_device(global_size=1 << 20, sm_count=4),
            frontend.build(),
            {"q": q, "k": k, "v": v, "mask": mask},
            [expected],
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(result.outputs[0], expected, rtol=1e-4, atol=1e-4)

    def test_large_linear_software_reference(self):
        rng = np.random.default_rng(23)
        x = rng.normal(size=(128, 256)).astype(np.float32)
        w = rng.normal(size=(512, 256)).astype(np.float32)
        bias = rng.normal(size=(512,)).astype(np.float32)
        frontend = ManualFrontend("large_linear")
        value = frontend.add_input("x", DType.F32, x.shape)
        weight = frontend.add_input("w", DType.F32, w.shape)
        b = frontend.add_input("b", DType.F32, bias.shape)
        frontend.output(frontend.linear(value, weight, b))
        expected = x @ w.T + bias
        result = run_software_model_e2e(
            make_device(global_size=1 << 24, sm_count=4),
            frontend.build(),
            {"x": x, "w": w, "b": bias},
            [expected],
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(result.outputs[0], expected, rtol=1e-4, atol=1e-4)

    def test_resize_software_matches_mgpu(self):
        image = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
        frontend = ManualFrontend("resize")
        x = frontend.add_input("x", DType.F32, image.shape)
        frontend.output(frontend.resize(x, sizes=(1, 1, 4, 4)))
        expected = image.repeat(2, axis=2).repeat(2, axis=3)
        self._assert_equivalent(frontend.build(), {"x": image}, expected)

    def test_slice_software_matches_mgpu(self):
        image = np.arange(20, dtype=np.float32).reshape(4, 5)
        frontend = ManualFrontend("slice")
        x = frontend.add_input("x", DType.F32, image.shape)
        frontend.output(frontend.slice(x, starts=(1, 1), ends=(4, 4), axes=(0, 1)))
        self._assert_equivalent(frontend.build(), {"x": image}, image[1:4, 1:4])

    def test_transpose_software_matches_mgpu(self):
        image = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        frontend = ManualFrontend("transpose")
        x = frontend.add_input("x", DType.F32, image.shape)
        frontend.output(frontend.transpose(x, (1, 0, 2)))
        self._assert_equivalent(
            frontend.build(), {"x": image}, np.transpose(image, (1, 0, 2))
        )

    def test_concat_software_matches_mgpu(self):
        a = np.arange(6, dtype=np.float32).reshape(2, 3)
        b = np.arange(6, 12, dtype=np.float32).reshape(2, 3)
        frontend = ManualFrontend("concat")
        x = frontend.add_input("a", DType.F32, a.shape)
        y = frontend.add_input("b", DType.F32, b.shape)
        frontend.output(frontend.concat(x, y, axis=0))
        self._assert_equivalent(
            frontend.build(), {"a": a, "b": b}, np.concatenate([a, b], axis=0)
        )

    def test_rope_software_matches_mgpu(self):
        x = np.arange(16, dtype=np.float32).reshape(1, 1, 2, 8)
        cos = np.ones((1, 1, 2, 4), dtype=np.float32)
        sin = np.ones((1, 1, 2, 4), dtype=np.float32)
        frontend = ManualFrontend("rope")
        value = frontend.add_input("x", DType.F32, x.shape)
        c = frontend.add_input("cos", DType.F32, cos.shape)
        s = frontend.add_input("sin", DType.F32, sin.shape)
        frontend.output(frontend.rope(value, c, s))
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = x0 * cos - x1 * sin
        out[..., 1::2] = x0 * sin + x1 * cos
        self._assert_equivalent(frontend.build(), {"x": x, "cos": cos, "sin": sin}, out)

    def test_softmax_software_matches_mgpu(self):
        logits = np.asarray([[1.0, 2.0, 3.0], [1.0, 0.0, -1.0]], dtype=np.float32)
        frontend = ManualFrontend("softmax")
        x = frontend.add_input("x", DType.F32, logits.shape)
        frontend.output(frontend.softmax(x, axis=-1))
        shifted = logits - logits.max(axis=-1, keepdims=True)
        expected = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
        self._assert_equivalent(frontend.build(), {"x": logits}, expected)

    def test_layer_norm_software_matches_mgpu(self):
        x = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        w = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        b = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        frontend = ManualFrontend("layer_norm")
        value = frontend.add_input("x", DType.F32, x.shape)
        weight = frontend.add_input("w", DType.F32, w.shape)
        bias = frontend.add_input("b", DType.F32, b.shape)
        frontend.output(frontend.layer_norm(value, weight, bias))
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        expected = (x - mean) / np.sqrt(var + 1e-5)
        self._assert_equivalent(frontend.build(), {"x": x, "w": w, "b": b}, expected)

    def test_rms_norm_software_matches_mgpu(self):
        x = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        w = np.asarray([1.0, 1.0], dtype=np.float32)
        frontend = ManualFrontend("rms_norm")
        value = frontend.add_input("x", DType.F32, x.shape)
        weight = frontend.add_input("w", DType.F32, w.shape)
        frontend.output(frontend.rms_norm(value, weight))
        rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)
        expected = x / rms
        self._assert_equivalent(frontend.build(), {"x": x, "w": w}, expected)

    def test_causal_mask_software_matches_mgpu(self):
        frontend = ManualFrontend("causal_mask")
        frontend.output(frontend.causal_mask(query_len=3, key_len=3, batch=1))
        expected = np.tril(np.ones((3, 3), dtype=np.float32)).reshape(1, 1, 3, 3)
        self._assert_equivalent(frontend.build(), {}, expected)


if __name__ == "__main__":
    unittest.main()
