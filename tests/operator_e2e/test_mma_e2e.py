import unittest

import numpy as np

from hw.device import create_device, launch
from hw.isa import parse_asm


class TestMMATileEndToEnd(unittest.TestCase):
    def test_2x2_mma_tile_matches_numpy(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        expected = a @ b

        device = create_device()
        device.global_memory.load_floats(0, a.reshape(-1).tolist())
        device.global_memory.load_floats(16, b.reshape(-1).tolist())
        device.global_memory.load_floats(32, [0.0] * 4)

        program = parse_asm("""
            MOV R0, 0
            MOV R1, 16
            MOV R2, 32
            MOV R3, 0
            LDG R4, [R0 + 0]
            LDG R5, [R1 + 0]
            MMA R3, R4, R5, R3
            LDG R4, [R0 + 4]
            LDG R5, [R1 + 8]
            MMA R3, R4, R5, R3
            STG [R2 + 0], R3
            MOV R3, 0
            LDG R4, [R0 + 0]
            LDG R5, [R1 + 4]
            MMA R3, R4, R5, R3
            LDG R4, [R0 + 4]
            LDG R5, [R1 + 12]
            MMA R3, R4, R5, R3
            STG [R2 + 4], R3
            MOV R3, 0
            LDG R4, [R0 + 8]
            LDG R5, [R1 + 0]
            MMA R3, R4, R5, R3
            LDG R4, [R0 + 12]
            LDG R5, [R1 + 8]
            MMA R3, R4, R5, R3
            STG [R2 + 8], R3
            MOV R3, 0
            LDG R4, [R0 + 8]
            LDG R5, [R1 + 4]
            MMA R3, R4, R5, R3
            LDG R4, [R0 + 12]
            LDG R5, [R1 + 12]
            MMA R3, R4, R5, R3
            STG [R2 + 12], R3
            EXIT
            """)
        launch(device, program, grid_dim=(1,), block_dim=(1,))

        actual = np.asarray(
            device.global_memory.read_floats(32, 4), dtype=np.float32
        ).reshape(2, 2)
        np.testing.assert_allclose(actual, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
