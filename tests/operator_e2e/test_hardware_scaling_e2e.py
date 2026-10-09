"""Tests for the first hardware-scaling milestones."""

from __future__ import annotations

import unittest

import numpy as np

from compiler.codegen import lower_kernel
from compiler.frontends.manual_frontend import ManualFrontend
from compiler.mlir import MLIRBuilder, run_mgpu_pipeline
from compiler.model_ir import DType
from hw.device import create_device, launch
from hw.isa import parse_asm
from runtime.driver import Driver

from tests.operator_e2e.helpers import run_operator_e2e


class TestHardwareScalingEndToEnd(unittest.TestCase):
    def test_wave_scheduler_runs_grid_larger_than_sm_count(self):
        device = create_device(global_size=1 << 20, sm_count=4)
        program = parse_asm("""
            MOV R0, ctaid.x
            MUL R0, R0, 32
            ADD R0, R0, tid.x
            MUL R1, R0, 4
            STG [R1], R0
            EXIT
            """)

        launch(device, program, grid_dim=(8,), block_dim=(32,))

        actual = np.asarray(
            [device.global_memory.read_f32(4 * i) for i in range(256)],
            dtype=np.float32,
        )
        expected = np.arange(256, dtype=np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_register_count_is_configurable(self):
        device = create_device(global_size=1 << 20, sm_count=1, register_count=8)
        program = parse_asm("""
            MOV R7, tid.x
            MUL R0, R7, 4
            STG [R0], R7
            EXIT
            """)

        launch(device, program, grid_dim=(1,), block_dim=(4,))

        self.assertEqual(device.register_count, 8)
        self.assertEqual(len(device.sms[0].warps[0].lanes[0].regs.regs), 8)
        self.assertEqual(
            [device.global_memory.read_f32(4 * i) for i in range(4)],
            [0.0, 1.0, 2.0, 3.0],
        )

    def test_typed_runtime_buffers_round_trip(self):
        driver = Driver(global_size=1 << 20, sm_count=4)

        i8 = driver.allocate((4,), alignment=4, dtype="i8")
        i8.write(np.asarray([1.0, -2.0, 3.0, -4.0], dtype=np.float32))
        np.testing.assert_array_equal(i8.read(), np.asarray([1, -2, 3, -4]))

        i4 = driver.allocate((4,), alignment=4, dtype="i4")
        i4.write(np.asarray([1.0, -2.0, 3.0, -4.0], dtype=np.float32))
        np.testing.assert_array_equal(i4.read(), np.asarray([1, -2, 3, -4]))

        f16 = driver.allocate((4,), alignment=4, dtype="f16")
        f16.write(np.asarray([0.0, 0.5, -1.0, 2.0], dtype=np.float32))
        np.testing.assert_allclose(
            f16.read(), np.asarray([0.0, 0.5, -1.0, 2.0], dtype=np.float32)
        )

    def test_large_elementwise_kernel_uses_waves(self):
        n = 128
        frontend = ManualFrontend("large_add")
        a = frontend.add_input("a", DType.F32, (n,))
        b = frontend.add_input("b", DType.F32, (n,))
        frontend.output(frontend.add(a, b))
        result = run_mgpu_pipeline(MLIRBuilder().build(frontend.build()))

        a_data = np.arange(n, dtype=np.float32)
        b_data = np.ones(n, dtype=np.float32)
        expected = a_data + b_data
        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, a_data.tolist())
        device.global_memory.load_floats(n * 4, b_data.tolist())

        launch(
            device,
            parse_asm(result.code),
            grid_dim=result.grid,
            block_dim=result.block,
        )

        actual = np.asarray(
            device.global_memory.read_floats(2 * n * 4, n), dtype=np.float32
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_large_reduction_conv_uses_loop_isa(self):
        rng = np.random.default_rng(8)
        image = rng.normal(size=(1, 2, 5, 5)).astype(np.float32)
        weight = rng.normal(size=(2, 2, 3, 3)).astype(np.float32)
        bias = rng.normal(size=(2,)).astype(np.float32)

        frontend = ManualFrontend("scalar_conv")
        x = frontend.add_input("x", DType.F32, image.shape)
        w = frontend.add_input("w", DType.F32, weight.shape)
        b = frontend.add_input("b", DType.F32, bias.shape)
        frontend.output(frontend.conv2d(x, w, b))

        def reference(image, weight, bias):
            n, c, h, w = image.shape
            f, _, kh, kw = weight.shape
            out = np.zeros((n, f, h - kh + 1, w - kw + 1), dtype=np.float32)
            for ff in range(f):
                for oy in range(out.shape[2]):
                    for ox in range(out.shape[3]):
                        acc = bias[ff]
                        for cc in range(c):
                            for ky in range(kh):
                                for kx in range(kw):
                                    acc += (
                                        image[0, cc, oy + ky, ox + kx]
                                        * weight[ff, cc, ky, kx]
                                    )
                        out[0, ff, oy, ox] = acc
            return out

        result = run_operator_e2e(
            create_device(global_size=1 << 20, sm_count=4),
            frontend.build(),
            {"x": image, "w": weight, "b": bias},
            [reference(image, weight, bias)],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertIn("conv_reduce_loop", result.isa_text)

    def test_runtime_launch_queue_executes_fifo(self):
        driver = Driver(global_size=1 << 20, sm_count=4)
        first = driver.allocate((32,), alignment=4)
        second = driver.allocate((32,), alignment=4)

        def kernel(base, delta):
            return parse_asm(f"""
                MOV R0, tid.x
                ADD R1, R0, {delta}
                MUL R2, R0, 4
                ADD R2, R2, {base}
                STG [R2], R1
                EXIT
                """)

        driver.enqueue(kernel(first.base_addr, 0), block_dim=(32,))
        driver.enqueue(kernel(second.base_addr, 100), block_dim=(32,))
        driver.synchronize()

        np.testing.assert_array_equal(first.read(), np.arange(32, dtype=np.float32))
        np.testing.assert_array_equal(
            second.read(), np.arange(32, dtype=np.float32) + 100.0
        )
        self.assertEqual(len(driver.profiler.records), 2)

    def test_large_scalar_matmul_uses_waves(self):
        rng = np.random.default_rng(9)
        a = rng.normal(size=(8, 4)).astype(np.float32)
        b = rng.normal(size=(4, 8)).astype(np.float32)
        expected = a @ b

        frontend = ManualFrontend("scalar_matmul")
        lhs = frontend.add_input("a", DType.F32, a.shape)
        rhs = frontend.add_input("b", DType.F32, b.shape)
        frontend.output(frontend.matmul(lhs, rhs))

        result = run_operator_e2e(
            create_device(global_size=1 << 20, sm_count=4),
            frontend.build(),
            {"a": a, "b": b},
            [expected],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertIn("ctaid.x", result.isa_text)
        self.assertIn("mat_loop", result.isa_text)

    def test_linear_with_bias_runs_in_simulator(self):
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        w = np.arange(20, dtype=np.float32).reshape(5, 4)
        b = np.arange(5, dtype=np.float32)
        expected = x @ w.T + b

        frontend = ManualFrontend("linear")
        value = frontend.add_input("x", DType.F32, x.shape)
        weight = frontend.add_input("w", DType.F32, w.shape)
        bias = frontend.add_input("b", DType.F32, b.shape)
        frontend.output(frontend.linear(value, weight, bias))

        result = run_operator_e2e(
            create_device(global_size=1 << 20, sm_count=4),
            frontend.build(),
            {"x": x, "w": w, "b": b},
            [expected],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertIn("linear_loop", result.isa_text)

    def test_gpt2_linear_shapes_lower_to_bounded_isa(self):
        for m, k, n in ((8, 768, 3072), (8, 3072, 768), (8, 768, 50257)):
            with self.subTest(m=m, k=k, n=n):
                frontend = ManualFrontend("gpt2_linear_lowering")
                value = frontend.add_input("x", DType.F32, (m, k))
                weight = frontend.add_input("w", DType.F32, (n, k))
                bias = frontend.add_input("b", DType.F32, (n,))
                frontend.output(frontend.linear(value, weight, bias))

                result = run_mgpu_pipeline(MLIRBuilder().build(frontend.build()))
                expected_grid = (m * n + 31) // 32
                self.assertEqual(result.grid, (expected_grid, 1, 1))
                self.assertIn("linear_loop", result.code)

    def test_masked_softmax_runs_in_simulator(self):
        logits = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]], np.float32)
        mask = np.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], np.float32)
        shifted = np.where(mask == 0, -1e30, logits)
        shifted = shifted - np.max(shifted, axis=1, keepdims=True)
        exp = np.exp(shifted)
        expected = exp / np.sum(exp, axis=1, keepdims=True)

        frontend = ManualFrontend("masked_softmax")
        x = frontend.add_input("x", DType.F32, logits.shape)
        m = frontend.add_input("mask", DType.F32, mask.shape)
        frontend.output(frontend.masked_softmax(x, m))

        result = run_operator_e2e(
            create_device(global_size=1 << 20, sm_count=4),
            frontend.build(),
            {"x": logits, "mask": mask},
            [expected],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertIn("mgpu.masked_softmax", result.mlir_text)


if __name__ == "__main__":
    unittest.main()
