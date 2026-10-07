import unittest

from hw.device import create_device, launch
from hw.isa import parse_asm

# A handwritten vector-add kernel.  Each thread handles one FP32 element using
# byte addresses: input A starts at 0, input B at 4096, and output at 8192.
VECTOR_ADD_ASM = """
MUL R0, tid, 4
LDG R1, [R0]
LDG R2, [R0 + 4096]
ADD R3, R1, R2
STG [R0 + 8192], R3
EXIT
"""


class TestVectorAddEndToEnd(unittest.TestCase):
    """Full ISA -> simulator integration test for elementwise vector add."""

    def test_vector_add(self):
        # Build tiny host-side inputs and a NumPy-free expected result.  This
        # test intentionally avoids NumPy to prove the simulator itself is
        # doing the computation.
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
        # The simple deterministic scheduler must produce the same cycle count
        # for identical launches.  This is the base contract for later
        # performance counters.
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
