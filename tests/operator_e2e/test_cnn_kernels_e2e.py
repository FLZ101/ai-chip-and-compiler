import unittest

import numpy as np

from compiler.codegen import lower_kernel
from compiler.frontends.manual_frontend import ManualFrontend
from compiler.mlir import MLIRBuilder, run_mgpu_pipeline
from compiler.model_ir import DType

from .helpers import make_device, run_operator_e2e


def _conv2d_reference(image, weight, bias, *, stride=1, padding=0, dilation=1):
    n, c, h, w = image.shape
    f, _, kh, kw = weight.shape
    effective_kh = dilation * (kh - 1) + 1
    effective_kw = dilation * (kw - 1) + 1
    out_h = (h + 2 * padding - effective_kh) // stride + 1
    out_w = (w + 2 * padding - effective_kw) // stride + 1
    output = np.zeros((n, f, out_h, out_w), dtype=np.float32)
    for nn in range(n):
        for ff in range(f):
            for oy in range(out_h):
                for ox in range(out_w):
                    acc = bias[ff]
                    for cc in range(c):
                        for ky in range(kh):
                            for kx in range(kw):
                                iy = oy * stride + ky * dilation - padding
                                ix = ox * stride + kx * dilation - padding
                                if 0 <= iy < h and 0 <= ix < w:
                                    acc += (
                                        image[nn, cc, iy, ix] * weight[ff, cc, ky, kx]
                                    )
                    output[nn, ff, oy, ox] = acc
    return output


def _depthwise_reference(image, weight, *, stride=1, padding=0):
    n, c, h, w = image.shape
    _, _, kh, kw = weight.shape
    out_h = (h + 2 * padding - kh) // stride + 1
    out_w = (w + 2 * padding - kw) // stride + 1
    output = np.zeros((n, c, out_h, out_w), dtype=np.float32)
    for nn in range(n):
        for cc in range(c):
            for oy in range(out_h):
                for ox in range(out_w):
                    acc = 0.0
                    for ky in range(kh):
                        for kx in range(kw):
                            iy = oy * stride + ky - padding
                            ix = ox * stride + kx - padding
                            if 0 <= iy < h and 0 <= ix < w:
                                acc += image[nn, cc, iy, ix] * weight[cc, 0, ky, kx]
                    output[nn, cc, oy, ox] = acc
    return output


def _pool_reference(image, kernel, stride, mode):
    n, c, h, w = image.shape
    kh, kw = kernel
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    output = np.zeros((n, c, out_h, out_w), dtype=np.float32)
    for nn in range(n):
        for cc in range(c):
            for oy in range(out_h):
                for ox in range(out_w):
                    window = image[
                        nn,
                        cc,
                        oy * stride : oy * stride + kh,
                        ox * stride : ox * stride + kw,
                    ]
                    if mode == "max":
                        output[nn, cc, oy, ox] = np.max(window)
                    else:
                        output[nn, cc, oy, ox] = np.mean(window)
    return output


class TestCNNkernelsEndToEnd(unittest.TestCase):
    """Step 010 CNN and pooling operator tests."""

    def test_conv2d(self):
        rng = np.random.default_rng(7)
        image = rng.normal(size=(1, 2, 4, 4)).astype(np.float32)
        weight = rng.normal(size=(2, 2, 2, 2)).astype(np.float32)
        bias = rng.normal(size=(2,)).astype(np.float32)

        frontend = ManualFrontend("conv2d")
        x = frontend.add_input("x", DType.F32, image.shape)
        w = frontend.add_input("w", DType.F32, weight.shape)
        b = frontend.add_input("b", DType.F32, bias.shape)
        frontend.output(frontend.conv2d(x, w, b))

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"x": image, "w": weight, "b": bias},
            [_conv2d_reference(image, weight, bias)],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertIn("MMA", result.isa_text)
        self.assertIn("IDIV", result.isa_text)

    def test_conv2d_stride_padding_dilation(self):
        rng = np.random.default_rng(21)
        image = rng.normal(size=(1, 2, 8, 8)).astype(np.float32)
        weight = rng.normal(size=(2, 2, 3, 3)).astype(np.float32)
        bias = rng.normal(size=(2,)).astype(np.float32)

        frontend = ManualFrontend("conv2d_dilated")
        x = frontend.add_input("x", DType.F32, image.shape)
        w = frontend.add_input("w", DType.F32, weight.shape)
        b = frontend.add_input("b", DType.F32, bias.shape)
        frontend.output(
            frontend.conv2d(
                x,
                w,
                b,
                strides=2,
                padding=1,
                dilation=2,
            )
        )

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"x": image, "w": weight, "b": bias},
            [
                _conv2d_reference(
                    image,
                    weight,
                    bias,
                    stride=2,
                    padding=1,
                    dilation=2,
                )
            ],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertIn("conv_reduce_loop", result.isa_text)

    def test_depthwise_conv2d(self):
        rng = np.random.default_rng(11)
        image = rng.normal(size=(1, 2, 4, 4)).astype(np.float32)
        weight = rng.normal(size=(2, 1, 2, 2)).astype(np.float32)

        frontend = ManualFrontend("depthwise")
        x = frontend.add_input("x", DType.F32, image.shape)
        w = frontend.add_input("w", DType.F32, weight.shape)
        frontend.output(frontend.depthwise_conv2d(x, w))

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"x": image, "w": weight},
            [_depthwise_reference(image, weight)],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertIn("FMA", result.isa_text)

    def test_max_pool2d(self):
        image = np.arange(1 * 2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4)
        frontend = ManualFrontend("max_pool")
        x = frontend.add_input("x", DType.F32, image.shape)
        frontend.output(frontend.max_pool2d(x, kernel=2, strides=2))

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"x": image},
            [_pool_reference(image, (2, 2), 2, "max")],
            lowering=lower_kernel,
        )
        self.assertIn("MAX", result.isa_text)

    def test_large_pool_kernel_uses_loop_isa(self):
        image = np.arange(1 * 2 * 8 * 8, dtype=np.float32).reshape(1, 2, 8, 8)
        frontend = ManualFrontend("large_max_pool")
        x = frontend.add_input("x", DType.F32, image.shape)
        frontend.output(frontend.max_pool2d(x, kernel=3, strides=2))

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"x": image},
            [_pool_reference(image, (3, 3), 2, "max")],
            lowering=lower_kernel,
        )
        self.assertIn("pool_loop", result.isa_text)

    def test_224_conv_compiles_to_bounded_loop_isa(self):
        frontend = ManualFrontend("conv224")
        x = frontend.add_input("x", DType.F32, (1, 3, 224, 224))
        w = frontend.add_input("w", DType.F32, (8, 3, 7, 7))
        b = frontend.add_input("b", DType.F32, (8,))
        frontend.output(frontend.conv2d(x, w, b, strides=2, padding=3))

        result = run_mgpu_pipeline(MLIRBuilder().build(frontend.build()))

        self.assertIn("conv_reduce_loop", result.code)
        self.assertLess(len(result.code.splitlines()), 200)
        self.assertEqual(result.grid, (3136, 1, 1))

    def test_avg_pool2d(self):
        image = np.arange(1 * 2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4)
        frontend = ManualFrontend("avg_pool")
        x = frontend.add_input("x", DType.F32, image.shape)
        frontend.output(frontend.avg_pool2d(x, kernel=2, strides=2))

        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"x": image},
            [_pool_reference(image, (2, 2), 2, "avg")],
            lowering=lower_kernel,
        )
        self.assertIn("ADD", result.isa_text)

    def test_global_avg_pool(self):
        image = np.arange(1 * 2 * 3 * 3, dtype=np.float32).reshape(1, 2, 3, 3)
        frontend = ManualFrontend("global_avg_pool")
        x = frontend.add_input("x", DType.F32, image.shape)
        frontend.output(frontend.global_avg_pool(x))

        expected = image.mean(axis=(2, 3), keepdims=True)
        result = run_operator_e2e(
            make_device(),
            frontend.build(),
            {"x": image},
            [expected],
            lowering=lower_kernel,
        )
        self.assertIn("DIV", result.isa_text)
        self.assertIn("gap_loop", result.isa_text)


if __name__ == "__main__":
    unittest.main()
