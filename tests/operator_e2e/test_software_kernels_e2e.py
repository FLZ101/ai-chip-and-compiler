"""Differential tests for native software operator kernels."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

from compiler.frontends.manual_frontend import ManualFrontend
from compiler.model_ir import DType

from .helpers import make_device, run_operator_e2e

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


if __name__ == "__main__":
    unittest.main()
