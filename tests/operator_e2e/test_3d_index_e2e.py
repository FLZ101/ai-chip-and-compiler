import unittest

from hw.device import create_device, launch
from hw.isa import parse_asm

THREE_DIM_INDEX_ASM = """
MOV R0, ctaid.z
MUL R0, R0, 2
ADD R0, R0, ctaid.y
MUL R0, R0, 2
ADD R0, R0, ctaid.x
MUL R0, R0, 8
MOV R1, tid.z
MUL R1, R1, 2
ADD R1, R1, tid.y
MUL R1, R1, 2
ADD R1, R1, tid.x
ADD R0, R0, R1
MOV R2, R0
MUL R0, R0, 4
STG [R0], R2
EXIT
"""


class Test3DIndexEndToEnd(unittest.TestCase):
    def test_3d_grid_and_block_indexing(self):
        device = create_device(global_size=1 << 20, sm_count=4)

        launch(
            device,
            parse_asm(THREE_DIM_INDEX_ASM),
            grid_dim=(2, 2, 1),
            block_dim=(2, 2, 2),
        )

        expected = [float(i) for i in range(32)]
        self.assertEqual(device.global_memory.read_floats(0, 32), expected)


if __name__ == "__main__":
    unittest.main()
