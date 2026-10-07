import unittest

from hw.device import create_device, launch
from hw.isa import parse_asm

VECTOR_ADD_ASM = """
MUL R0, tid, 4
LDG R1, [R0]
LDG R2, [R0 + 4096]
ADD R3, R1, R2
STG [R0 + 8192], R3
EXIT
"""


class TestVectorAddEndToEnd(unittest.TestCase):
    def test_vector_add(self):
        n = 8
        a = [float(i + 1) for i in range(n)]
        b = [float(10 * i) for i in range(n)]
        expected = [x + y for x, y in zip(a, b)]

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, a)
        device.global_memory.load_floats(4096, b)

        program = parse_asm(VECTOR_ADD_ASM)
        launch(device, program, grid_dim=(1,), block_dim=(n,))

        actual = device.global_memory.read_floats(8192, n)
        self.assertEqual(actual, expected)

    def test_simulator_is_cycle_stable(self):
        cycles = set()
        for _ in range(3):
            device = create_device(global_size=1 << 20, sm_count=4)
            device.global_memory.load_floats(0, [1.0, 2.0, 3.0, 4.0])
            device.global_memory.load_floats(4096, [4.0, 3.0, 2.0, 1.0])
            launch(device, parse_asm(VECTOR_ADD_ASM), grid_dim=(1,), block_dim=(4,))
            cycles.add(device.clock)

        self.assertEqual(len(cycles), 1)


if __name__ == "__main__":
    unittest.main()
