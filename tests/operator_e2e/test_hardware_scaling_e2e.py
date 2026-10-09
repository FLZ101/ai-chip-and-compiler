"""Tests for the first hardware-scaling milestones."""

from __future__ import annotations

import unittest

import numpy as np

from compiler.frontends.manual_frontend import ManualFrontend
from compiler.mlir import MLIRBuilder, run_mgpu_pipeline
from compiler.model_ir import DType
from hw.device import create_device, launch
from hw.isa import parse_asm
from runtime.driver import Driver


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


if __name__ == "__main__":
    unittest.main()
