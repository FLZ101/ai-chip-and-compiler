import unittest

from hw.device import create_device, launch
from hw.isa import parse_asm


class TestTypedMemoryEndToEnd(unittest.TestCase):
    def test_f16_store_executes_in_simulator(self):
        device = create_device(global_size=1 << 20, sm_count=4)
        launch(
            device,
            parse_asm("""
                MOV R0, tid.x
                MUL R0, R0, 2
                MOV R1, tid.x
                STG.F16 [R0], R1
                EXIT
                """),
            grid_dim=(1,),
            block_dim=(4,),
        )

        self.assertEqual(
            [device.global_memory.read_f16(addr) for addr in (0, 2, 4, 6)],
            [0.0, 1.0, 2.0, 3.0],
        )

    def test_i8_store_executes_in_simulator(self):
        device = create_device(global_size=1 << 20, sm_count=4)
        launch(
            device,
            parse_asm("""
                MOV R0, tid.x
                MOV R1, tid.x
                STG.I8 [R0], R1
                EXIT
                """),
            grid_dim=(1,),
            block_dim=(4,),
        )

        self.assertEqual(
            [device.global_memory.read_i8(addr) for addr in range(4)],
            [0, 1, 2, 3],
        )

    def test_i4_packed_store_executes_in_simulator(self):
        device = create_device(global_size=1 << 20, sm_count=4)
        launch(
            device,
            parse_asm("""
                MOV R0, tid.x
                MOV R1, tid.x
                STG.I4 [R0], R1
                EXIT
                """),
            grid_dim=(1,),
            block_dim=(4,),
        )

        self.assertEqual(
            [device.global_memory.read_i4(addr) for addr in range(4)],
            [0, 1, 2, 3],
        )


if __name__ == "__main__":
    unittest.main()
