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


if __name__ == "__main__":
    unittest.main()
