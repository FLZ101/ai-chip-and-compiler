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

    def _run_activation_via_mgpu_opt(self, op_name: str, n: int):
        frontend = ManualFrontend(op_name)
        x = frontend.add_input("x", DType.F32, (n,))
        if op_name == "relu":
            output = frontend.relu(x)
        elif op_name == "leaky_relu":
            output = frontend.leaky_relu(x)
        elif op_name == "silu":
            output = frontend.silu(x)
        elif op_name == "gelu":
            output = frontend.gelu(x)
        else:
            output = frontend.sigmoid(x)
        frontend.output(output)
        graph = frontend.build()

        result = run_mgpu_pipeline(MLIRBuilder().build(graph))
        data = np.linspace(-2.0, 2.0, n, dtype=np.float32)
        if op_name == "relu":
            expected = np.maximum(data, 0.0)
        elif op_name == "leaky_relu":
            expected = np.where(data < 0.0, 0.01 * data, data)
        elif op_name == "silu":
            expected = data / (1.0 + np.exp(-data))
        elif op_name == "gelu":
            expected = (
                0.5
                * data
                * (1.0 + np.tanh(0.7978845608028654 * (data + 0.044715 * data**3)))
            )
        else:
            expected = 1.0 / (1.0 + np.exp(-data))

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, data.tolist())
        device.global_memory.load_floats(n * 4, [0.0] * n)

        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(n * 4, n), dtype=np.float32
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_activations_use_mgpu_opt(self):
        for op_name in ("relu", "leaky_relu", "silu", "gelu", "sigmoid"):
            with self.subTest(op=op_name):
                self._run_activation_via_mgpu_opt(op_name, 16)

    @staticmethod
    def _conv2d_reference(image, weight):
        n, c, h, w = image.shape
        f, _, kh, kw = weight.shape
        out_h = h - kh + 1
        out_w = w - kw + 1
        output = np.zeros((n, f, out_h, out_w), dtype=np.float32)
        for ff in range(f):
            for oy in range(out_h):
                for ox in range(out_w):
                    acc = 0.0
                    for cc in range(c):
                        for ky in range(kh):
                            for kx in range(kw):
                                acc += (
                                    image[0, cc, oy + ky, ox + kx]
                                    * weight[ff, cc, ky, kx]
                                )
                    output[0, ff, oy, ox] = acc
        return output

    @staticmethod
    def _depthwise_reference(image, weight):
        n, c, h, w = image.shape
        _, _, kh, kw = weight.shape
        out_h = h - kh + 1
        out_w = w - kw + 1
        output = np.zeros((n, c, out_h, out_w), dtype=np.float32)
        for cc in range(c):
            for oy in range(out_h):
                for ox in range(out_w):
                    acc = 0.0
                    for ky in range(kh):
                        for kx in range(kw):
                            acc += (
                                image[0, cc, oy + ky, ox + kx] * weight[cc, 0, ky, kx]
                            )
                    output[0, cc, oy, ox] = acc
        return output

    def test_conv2d_uses_mgpu_opt(self):
        rng = np.random.default_rng(21)
        image = rng.normal(size=(1, 2, 4, 4)).astype(np.float32)
        weight = rng.normal(size=(2, 2, 2, 2)).astype(np.float32)
        frontend = ManualFrontend("conv2d")
        x = frontend.add_input("x", DType.F32, image.shape)
        w = frontend.add_input("w", DType.F32, weight.shape)
        frontend.output(frontend.conv2d(x, w))
        result = run_mgpu_pipeline(MLIRBuilder().build(frontend.build()))

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, image.reshape(-1).tolist())
        device.global_memory.load_floats(image.nbytes, weight.reshape(-1).tolist())
        device.global_memory.load_floats(image.nbytes + weight.nbytes, [0.0] * 18)
        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(image.nbytes + weight.nbytes, 18),
            dtype=np.float32,
        ).reshape(1, 2, 3, 3)
        np.testing.assert_allclose(
            actual, self._conv2d_reference(image, weight), rtol=1e-4, atol=1e-5
        )

    def test_depthwise_conv2d_uses_mgpu_opt(self):
        rng = np.random.default_rng(22)
        image = rng.normal(size=(1, 2, 4, 4)).astype(np.float32)
        weight = rng.normal(size=(2, 1, 2, 2)).astype(np.float32)
        frontend = ManualFrontend("depthwise")
        x = frontend.add_input("x", DType.F32, image.shape)
        w = frontend.add_input("w", DType.F32, weight.shape)
        frontend.output(frontend.depthwise_conv2d(x, w))
        result = run_mgpu_pipeline(MLIRBuilder().build(frontend.build()))

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, image.reshape(-1).tolist())
        weight_base = image.nbytes
        device.global_memory.load_floats(weight_base, weight.reshape(-1).tolist())
        output_base = weight_base + weight.nbytes
        device.global_memory.load_floats(output_base, [0.0] * 18)
        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(output_base, 18), dtype=np.float32
        ).reshape(1, 2, 3, 3)
        np.testing.assert_allclose(
            actual,
            self._depthwise_reference(image, weight),
            rtol=1e-4,
            atol=1e-5,
        )

    def _run_pool_via_mgpu_opt(self, op_name):
        image = np.arange(1 * 2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4)
        frontend = ManualFrontend(op_name)
        x = frontend.add_input("x", DType.F32, image.shape)
        if op_name == "max_pool":
            output = frontend.max_pool2d(x, kernel=2, strides=2)
        else:
            output = frontend.avg_pool2d(x, kernel=2, strides=2)
        frontend.output(output)
        result = run_mgpu_pipeline(MLIRBuilder().build(frontend.build()))

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, image.reshape(-1).tolist())
        device.global_memory.load_floats(image.nbytes, [0.0] * 8)
        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(image.nbytes, 8), dtype=np.float32
        ).reshape(1, 2, 2, 2)
        if op_name == "max_pool":
            expected = image.reshape(1, 2, 2, 2, 2, 2).max(axis=(3, 5))
        else:
            expected = image.reshape(1, 2, 2, 2, 2, 2).mean(axis=(3, 5))
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_pooling_uses_mgpu_opt(self):
        for op_name in ("max_pool", "avg_pool"):
            with self.subTest(op=op_name):
                self._run_pool_via_mgpu_opt(op_name)

    def test_global_avg_pool_uses_mgpu_opt(self):
        image = np.arange(1 * 2 * 3 * 3, dtype=np.float32).reshape(1, 2, 3, 3)
        frontend = ManualFrontend("gap")
        x = frontend.add_input("x", DType.F32, image.shape)
        frontend.output(frontend.global_avg_pool(x))
        result = run_mgpu_pipeline(MLIRBuilder().build(frontend.build()))

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, image.reshape(-1).tolist())
        device.global_memory.load_floats(image.nbytes, [0.0] * 2)
        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(image.nbytes, 2), dtype=np.float32
        ).reshape(1, 2, 1, 1)
        expected = image.mean(axis=(2, 3), keepdims=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    @staticmethod
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

    def test_rope_uses_mgpu_opt(self):
        x = np.arange(8, dtype=np.float32).reshape(2, 4)
        cos = np.array([[0.9, 0.9, 0.8, 0.8], [0.7, 0.7, 0.6, 0.6]], dtype=np.float32)
        sin = np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]], dtype=np.float32)
        frontend = ManualFrontend("rope")
        xv = frontend.add_input("x", DType.F32, x.shape)
        cv = frontend.add_input("cos", DType.F32, cos.shape)
        sv = frontend.add_input("sin", DType.F32, sin.shape)
        frontend.output(frontend.rope(xv, cv, sv))
        result = run_mgpu_pipeline(MLIRBuilder().build(frontend.build()))

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, x.reshape(-1).tolist())
        device.global_memory.load_floats(x.nbytes, cos.reshape(-1).tolist())
        device.global_memory.load_floats(
            x.nbytes + cos.nbytes, sin.reshape(-1).tolist()
        )
        output_base = x.nbytes + cos.nbytes + sin.nbytes
        device.global_memory.load_floats(output_base, [0.0] * x.size)
        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(output_base, x.size), dtype=np.float32
        ).reshape(x.shape)
        np.testing.assert_allclose(
            actual, self._rope_reference(x, cos, sin), rtol=1e-4, atol=1e-5
        )

    def test_causal_mask_uses_mgpu_opt(self):
        frontend = ManualFrontend("causal_mask")
        frontend.output(frontend.causal_mask(query_len=3, key_len=3, batch=1))
        result = run_mgpu_pipeline(MLIRBuilder().build(frontend.build()))

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, [0.0] * 9)
        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(0, 9), dtype=np.float32
        ).reshape(1, 1, 3, 3)
        expected = np.tril(np.ones((3, 3), dtype=np.float32)).reshape(1, 1, 3, 3)
        np.testing.assert_array_equal(actual, expected)

    def test_kv_cache_append_uses_mgpu_opt(self):
        cache = np.zeros((1, 1, 2, 2), dtype=np.float32)
        value = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
        frontend = ManualFrontend("kv_append")
        c = frontend.add_input("cache", DType.F32, cache.shape)
        v = frontend.add_input("value", DType.F32, value.shape)
        frontend.output(frontend.kv_cache_append(c, v))
        result = run_mgpu_pipeline(MLIRBuilder().build(frontend.build()))

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, cache.reshape(-1).tolist())
        device.global_memory.load_floats(cache.nbytes, value.reshape(-1).tolist())
        output_base = cache.nbytes + value.nbytes
        device.global_memory.load_floats(output_base, [0.0] * value.size)
        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(output_base, value.size), dtype=np.float32
        ).reshape(value.shape)
        np.testing.assert_array_equal(actual, value)

    def test_kv_cache_gather_uses_mgpu_opt(self):
        cache = np.arange(8, dtype=np.float32).reshape(1, 1, 4, 2)
        positions = np.array([[2]], dtype=np.float32)
        frontend = ManualFrontend("kv_gather")
        c = frontend.add_input("cache", DType.F32, cache.shape)
        p = frontend.add_input("positions", DType.F32, positions.shape)
        frontend.output(frontend.kv_cache_gather(c, p))
        result = run_mgpu_pipeline(MLIRBuilder().build(frontend.build()))

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, cache.reshape(-1).tolist())
        device.global_memory.load_floats(cache.nbytes, positions.reshape(-1).tolist())
        output_base = cache.nbytes + positions.nbytes
        device.global_memory.load_floats(output_base, [0.0] * 2)
        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(output_base, 2), dtype=np.float32
        ).reshape(1, 1, 1, 2)
        np.testing.assert_array_equal(actual, cache[:, :, 2:3, :])

    @staticmethod
    def _softmax_reference(x):
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    @staticmethod
    def _attention_reference(q, k, v, scale, mask=None):
        scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
        if mask is not None:
            scores = np.where(mask == 0, -np.inf, scores)
        weights = TestMgpuOptPipelineEndToEnd._softmax_reference(scores)
        return np.matmul(weights, v)

    def test_attention_uses_mgpu_opt(self):
        q = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
        k = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        v = np.array([[[[2.0, 1.0], [0.0, 3.0]]]], dtype=np.float32)
        mask = np.array([[[[1.0, 0.0], [1.0, 1.0]]]], dtype=np.float32)
        scale = 1.0 / np.sqrt(2.0)
        mlir = f"""
module {{
  func.func @attention(%q: memref<2x2xf32>, %k: memref<2x2xf32>, %v: memref<2x2xf32>, %mask: memref<2x2xf32>, %out: memref<2x2xf32>) {{
    %c1 = arith.constant 1 : index
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) {{
      mgpu.attention %q, %k, %v, %mask, %out {{scale = {scale} : f32, masked}} : memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>
      mgpu.return
    }}
    return
  }}
}}
"""
        result = run_mgpu_pipeline(mlir)
        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, q.reshape(-1).tolist())
        device.global_memory.load_floats(16, k.reshape(-1).tolist())
        device.global_memory.load_floats(32, v.reshape(-1).tolist())
        device.global_memory.load_floats(48, mask.reshape(-1).tolist())
        device.global_memory.load_floats(64, [0.0] * 4)
        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(64, 4), dtype=np.float32
        ).reshape(1, 1, 2, 2)
        np.testing.assert_allclose(
            actual,
            self._attention_reference(q, k, v, scale, mask),
            rtol=1e-4,
            atol=1e-5,
        )

    @staticmethod
    def _softmax_reference_2d(x):
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def test_softmax_uses_mgpu_opt(self):
        x = np.array(
            [[1.0, 2.0, 3.0, 4.0], [1000.0, 1000.0, 1000.0, 1000.0]],
            dtype=np.float32,
        )
        mlir = """
module {
  func.func @softmax(%x: memref<2x4xf32>, %out: memref<2x4xf32>) {
    %c1 = arith.constant 1 : index
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) {
      mgpu.softmax %x, %out : memref<2x4xf32>, memref<2x4xf32>
      mgpu.return
    }
    return
  }
}
"""
        result = run_mgpu_pipeline(mlir)
        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, x.reshape(-1).tolist())
        device.global_memory.load_floats(x.nbytes, [0.0] * x.size)
        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(x.nbytes, x.size), dtype=np.float32
        ).reshape(x.shape)
        np.testing.assert_allclose(
            actual, self._softmax_reference_2d(x), rtol=1e-4, atol=1e-5
        )

    def test_layer_norm_uses_mgpu_opt(self):
        x = np.arange(16, dtype=np.float32).reshape(4, 4)
        weight = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float32)
        bias = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
        mlir = """
module {
  func.func @layernorm(%x: memref<4x4xf32>, %w: memref<4xf32>, %b: memref<4xf32>, %out: memref<4x4xf32>) {
    %c1 = arith.constant 1 : index
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) {
      mgpu.layer_norm %x, %w, %b, %out : memref<4x4xf32>, memref<4xf32>, memref<4xf32>, memref<4x4xf32>
      mgpu.return
    }
    return
  }
}
"""
        result = run_mgpu_pipeline(mlir)
        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, x.reshape(-1).tolist())
        device.global_memory.load_floats(x.nbytes, weight.tolist())
        device.global_memory.load_floats(x.nbytes + weight.nbytes, bias.tolist())
        output_base = x.nbytes + weight.nbytes + bias.nbytes
        device.global_memory.load_floats(output_base, [0.0] * x.size)
        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(output_base, x.size), dtype=np.float32
        ).reshape(x.shape)
        mean = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1, keepdims=True)
        expected = (x - mean) / np.sqrt(var + 1e-5) * weight + bias
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_rms_norm_uses_mgpu_opt(self):
        x = np.arange(16, dtype=np.float32).reshape(4, 4)
        weight = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
        mlir = """
module {
  func.func @rmsnorm(%x: memref<4x4xf32>, %w: memref<4xf32>, %out: memref<4x4xf32>) {
    %c1 = arith.constant 1 : index
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) {
      mgpu.rms_norm %x, %w, %out : memref<4x4xf32>, memref<4xf32>, memref<4x4xf32>
      mgpu.return
    }
    return
  }
}
"""
        result = run_mgpu_pipeline(mlir)
        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, x.reshape(-1).tolist())
        device.global_memory.load_floats(x.nbytes, weight.tolist())
        output_base = x.nbytes + weight.nbytes
        device.global_memory.load_floats(output_base, [0.0] * x.size)
        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(output_base, x.size), dtype=np.float32
        ).reshape(x.shape)
        rms = np.sqrt(np.mean(x * x, axis=1, keepdims=True) + 1e-5)
        expected = x / rms * weight
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_reshape_uses_mgpu_opt(self):
        x = np.arange(16, dtype=np.float32)
        frontend = ManualFrontend("reshape")
        value = frontend.add_input("x", DType.F32, x.shape)
        frontend.output(frontend.reshape(value, (4, 4)))
        result = run_mgpu_pipeline(MLIRBuilder().build(frontend.build()))

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, x.tolist())
        device.global_memory.load_floats(x.nbytes, [0.0] * x.size)
        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(x.nbytes, x.size), dtype=np.float32
        ).reshape(4, 4)
        np.testing.assert_array_equal(actual, x.reshape(4, 4))

    def test_transpose_uses_mgpu_opt(self):
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        frontend = ManualFrontend("transpose")
        value = frontend.add_input("x", DType.F32, x.shape)
        frontend.output(frontend.transpose(value, (1, 0)))
        result = run_mgpu_pipeline(MLIRBuilder().build(frontend.build()))

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, x.reshape(-1).tolist())
        device.global_memory.load_floats(x.nbytes, [0.0] * x.size)
        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(x.nbytes, x.size), dtype=np.float32
        ).reshape(3, 2)
        np.testing.assert_array_equal(actual, x.T)

    def test_concat_uses_mgpu_opt(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        frontend = ManualFrontend("concat")
        av = frontend.add_input("a", DType.F32, a.shape)
        bv = frontend.add_input("b", DType.F32, b.shape)
        frontend.output(frontend.concat(av, bv, axis=0))
        result = run_mgpu_pipeline(MLIRBuilder().build(frontend.build()))

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, a.tolist())
        device.global_memory.load_floats(a.nbytes, b.tolist())
        output_base = a.nbytes + b.nbytes
        device.global_memory.load_floats(output_base, [0.0] * (a.size + b.size))
        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(output_base, a.size + b.size),
            dtype=np.float32,
        )
        np.testing.assert_array_equal(actual, np.concatenate([a, b]))

    def test_slice_uses_mgpu_opt(self):
        x = np.arange(8, dtype=np.float32)
        frontend = ManualFrontend("slice")
        value = frontend.add_input("x", DType.F32, x.shape)
        frontend.output(frontend.slice(value, (2,), (6,), steps=(1,)))
        result = run_mgpu_pipeline(MLIRBuilder().build(frontend.build()))

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, x.tolist())
        device.global_memory.load_floats(x.nbytes, [0.0] * 4)
        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(x.nbytes, 4), dtype=np.float32
        )
        np.testing.assert_array_equal(actual, x[2:6])

    def test_pad_uses_mgpu_opt(self):
        x = np.arange(4, dtype=np.float32)
        frontend = ManualFrontend("pad")
        value = frontend.add_input("x", DType.F32, x.shape)
        frontend.output(frontend.pad(value, ((1, 2),), value=-1.0))
        result = run_mgpu_pipeline(MLIRBuilder().build(frontend.build()))

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, x.tolist())
        device.global_memory.load_floats(x.nbytes, [0.0] * 7)
        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(x.nbytes, 7), dtype=np.float32
        )
        np.testing.assert_array_equal(actual, np.pad(x, (1, 2), constant_values=-1.0))

    def test_resize_uses_mgpu_opt(self):
        x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        frontend = ManualFrontend("resize")
        value = frontend.add_input("x", DType.F32, x.shape)
        frontend.output(frontend.resize(value, sizes=(8,)))
        result = run_mgpu_pipeline(MLIRBuilder().build(frontend.build()))

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, x.tolist())
        device.global_memory.load_floats(x.nbytes, [0.0] * 8)
        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(x.nbytes, 8), dtype=np.float32
        )
        np.testing.assert_array_equal(actual, np.repeat(x, 2))

    def test_batch_norm_uses_mgpu_opt(self):
        x = np.arange(1 * 2 * 2 * 2, dtype=np.float32).reshape(1, 2, 2, 2)
        scale = np.array([1.0, 2.0], dtype=np.float32)
        bias = np.array([0.5, -0.5], dtype=np.float32)
        mean = np.array([1.0, 2.0], dtype=np.float32)
        var = np.array([0.5, 1.0], dtype=np.float32)
        frontend = ManualFrontend("batch_norm")
        xv = frontend.add_input("x", DType.F32, x.shape)
        sv = frontend.add_input("s", DType.F32, scale.shape)
        bv = frontend.add_input("b", DType.F32, bias.shape)
        mv = frontend.add_input("m", DType.F32, mean.shape)
        vv = frontend.add_input("v", DType.F32, var.shape)
        frontend.output(frontend.batch_norm(xv, sv, bv, mv, vv))
        result = run_mgpu_pipeline(MLIRBuilder().build(frontend.build()))

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, x.reshape(-1).tolist())
        device.global_memory.load_floats(x.nbytes, scale.tolist())
        device.global_memory.load_floats(x.nbytes + scale.nbytes, bias.tolist())
        device.global_memory.load_floats(
            x.nbytes + scale.nbytes + bias.nbytes, mean.tolist()
        )
        device.global_memory.load_floats(
            x.nbytes + scale.nbytes + bias.nbytes + mean.nbytes, var.tolist()
        )
        output_base = x.nbytes + scale.nbytes + bias.nbytes + mean.nbytes + var.nbytes
        device.global_memory.load_floats(output_base, [0.0] * x.size)
        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(output_base, x.size), dtype=np.float32
        ).reshape(x.shape)
        expected = (x - mean.reshape(1, -1, 1, 1)) / np.sqrt(
            var.reshape(1, -1, 1, 1) + 1e-5
        ) * scale.reshape(1, -1, 1, 1) + bias.reshape(1, -1, 1, 1)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_group_norm_uses_mgpu_opt(self):
        x = np.arange(1 * 2 * 2 * 2, dtype=np.float32).reshape(1, 2, 2, 2)
        weight = np.array([1.0, 2.0], dtype=np.float32)
        bias = np.array([0.0, 1.0], dtype=np.float32)
        frontend = ManualFrontend("group_norm")
        xv = frontend.add_input("x", DType.F32, x.shape)
        wv = frontend.add_input("w", DType.F32, weight.shape)
        bv = frontend.add_input("b", DType.F32, bias.shape)
        frontend.output(frontend.group_norm(xv, wv, bv, num_groups=2))
        result = run_mgpu_pipeline(MLIRBuilder().build(frontend.build()))

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, x.reshape(-1).tolist())
        device.global_memory.load_floats(x.nbytes, weight.tolist())
        device.global_memory.load_floats(x.nbytes + weight.nbytes, bias.tolist())
        output_base = x.nbytes + weight.nbytes + bias.nbytes
        device.global_memory.load_floats(output_base, [0.0] * x.size)
        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(output_base, x.size), dtype=np.float32
        ).reshape(x.shape)

        channels_per_group = x.shape[1] // 2
        expected = np.zeros_like(x)
        for group in range(2):
            start = group * channels_per_group
            end = start + channels_per_group
            group_x = x[:, start:end, :, :]
            mean = group_x.mean()
            var = group_x.var()
            expected[:, start:end, :, :] = (group_x - mean) / np.sqrt(
                var + 1e-5
            ) * weight[start:end].reshape(1, -1, 1, 1) + bias[start:end].reshape(
                1, -1, 1, 1
            )
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_box_decode_uses_mgpu_opt(self):
        boxes = np.array(
            [[10.0, 20.0, 4.0, 6.0], [30.0, 40.0, 2.0, 8.0]], dtype=np.float32
        )
        deltas = np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.1, -0.1, 0.2, -0.2]], dtype=np.float32
        )
        frontend = ManualFrontend("box_decode")
        bv = frontend.add_input("boxes", DType.F32, boxes.shape)
        dv = frontend.add_input("deltas", DType.F32, deltas.shape)
        frontend.output(frontend.box_decode(bv, dv))
        result = run_mgpu_pipeline(MLIRBuilder().build(frontend.build()))

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, boxes.reshape(-1).tolist())
        device.global_memory.load_floats(boxes.nbytes, deltas.reshape(-1).tolist())
        output_base = boxes.nbytes + deltas.nbytes
        device.global_memory.load_floats(output_base, [0.0] * boxes.size)
        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(output_base, boxes.size),
            dtype=np.float32,
        ).reshape(boxes.shape)
        cx = boxes[:, 0] + deltas[:, 0] * boxes[:, 2]
        cy = boxes[:, 1] + deltas[:, 1] * boxes[:, 3]
        w = boxes[:, 2] * np.exp(deltas[:, 2])
        h = boxes[:, 3] * np.exp(deltas[:, 3])
        expected = np.stack(
            [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=1
        ).astype(np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_nms_uses_mgpu_opt(self):
        boxes = np.array(
            [[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0], [5.0, 5.0, 7.0, 7.0]],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        frontend = ManualFrontend("nms")
        bv = frontend.add_input("boxes", DType.F32, boxes.shape)
        sv = frontend.add_input("scores", DType.F32, scores.shape)
        frontend.output(frontend.nms(bv, sv, iou_threshold=0.5))
        result = run_mgpu_pipeline(MLIRBuilder().build(frontend.build()))

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, boxes.reshape(-1).tolist())
        device.global_memory.load_floats(boxes.nbytes, scores.tolist())
        output_base = boxes.nbytes + scores.nbytes
        device.global_memory.load_floats(output_base, [0.0] * len(scores))
        launch(
            device, parse_asm(result.code), grid_dim=result.grid, block_dim=result.block
        )
        actual = np.asarray(
            device.global_memory.read_floats(output_base, len(scores)),
            dtype=np.float32,
        )

        def iou(a, b):
            x1 = max(a[0], b[0])
            y1 = max(a[1], b[1])
            x2 = min(a[2], b[2])
            y2 = min(a[3], b[3])
            inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
            area_a = (a[2] - a[0]) * (a[3] - a[1])
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            return inter / max(area_a + area_b - inter, 1.0)

        keep = np.ones(len(boxes), dtype=np.float32)
        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if scores[j] > scores[i] and iou(boxes[i], boxes[j]) > 0.5:
                    keep[i] = 0.0
        expected = np.where(keep, np.arange(len(boxes), dtype=np.float32), -1.0).astype(
            np.float32
        )
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
