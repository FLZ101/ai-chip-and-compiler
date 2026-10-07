import unittest

from hw.device import create_device, launch
from hw.isa import parse_asm

MULTI_WARP_VECTOR_ADD_ASM = """
MUL R0, tid.x, 4
LDG R1, [R0]
LDG R2, [R0 + 16384]
ADD R3, R1, R2
STG [R0 + 32768], R3
EXIT
"""


SHARED_MEMORY_BARRIER_ASM = """
MUL R0, tid.x, 4
STS [R0], tid.x
BAR.SYNC
MOV R1, ntid.x
SUB R2, R1, tid.x
SUB R2, R2, 1
MUL R2, R2, 4
LDS R3, [R2]
STG [R0], R3
EXIT
"""


class TestBarrierAndMultiWarpEndToEnd(unittest.TestCase):
    """Step 003 tests for multi-warp execution and block barriers."""

    def test_two_warp_vector_add(self):
        n = 64
        a = [float(i + 1) for i in range(n)]
        b = [float(i * 2) for i in range(n)]
        expected = [x + y for x, y in zip(a, b)]

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, a)
        device.global_memory.load_floats(16384, b)

        launch(
            device,
            parse_asm(MULTI_WARP_VECTOR_ADD_ASM),
            grid_dim=(1,),
            block_dim=(n,),
        )

        self.assertEqual(device.global_memory.read_floats(32768, n), expected)

    def test_shared_memory_barrier_across_warps(self):
        n = 64
        device = create_device(
            global_size=1 << 20,
            shared_size=1 << 16,
            sm_count=4,
        )

        launch(
            device,
            parse_asm(SHARED_MEMORY_BARRIER_ASM),
            grid_dim=(1,),
            block_dim=(n,),
        )

        expected = [float(i) for i in reversed(range(n))]
        self.assertEqual(device.global_memory.read_floats(0, n), expected)

    def test_multi_warp_barrier_is_deterministic(self):
        cycles = set()
        for _ in range(3):
            device = create_device(
                global_size=1 << 20,
                shared_size=1 << 16,
                sm_count=4,
            )
            launch(
                device,
                parse_asm(SHARED_MEMORY_BARRIER_ASM),
                grid_dim=(1,),
                block_dim=(64,),
            )
            cycles.add(device.clock)

        self.assertEqual(len(cycles), 1)


if __name__ == "__main__":
    unittest.main()
