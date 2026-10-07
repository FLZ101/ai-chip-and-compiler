import unittest

from hw.device import create_device, launch
from hw.isa import parse_asm

IF_ELSE_ASM = """
MUL R4, tid, 4
MOV R0, tid
MOV R1, 4
CMP.LT R2, R0, R1
BRA else, R2, end
MOV R3, 10
STG [R4], R3
BRA end
else:
MOV R3, 20
STG [R4], R3
BRA end
end:
EXIT
"""


NESTED_DIVERGENCE_ASM = """
MUL R4, tid, 4
MOV R0, tid
MOV R1, 6
CMP.LT R2, R0, R1
BRA inner, R2, end_outer
MOV R5, 3
STG [R4], R5
BRA end_outer
inner:
MOV R1, 2
CMP.LT R3, R0, R1
BRA else_inner, R3, end_inner
MOV R5, 1
STG [R4], R5
BRA end_inner
else_inner:
MOV R5, 2
STG [R4], R5
BRA end_inner
end_inner:
BRA end_outer
end_outer:
EXIT
"""


class TestDivergenceEndToEnd(unittest.TestCase):
    """SIMT branch divergence and reconvergence tests."""

    def test_if_else_divergence(self):
        device = create_device()
        program = parse_asm(IF_ELSE_ASM)
        launch(device, program, grid_dim=(1,), block_dim=(8,))

        expected = [20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0, 10.0]
        self.assertEqual(device.global_memory.read_floats(0, 8), expected)

    def test_nested_divergence(self):
        device = create_device()
        program = parse_asm(NESTED_DIVERGENCE_ASM)
        launch(device, program, grid_dim=(1,), block_dim=(8,))

        # The inner branch takes lanes 0 and 1 to else_inner (value 2), lanes
        # 2..5 to the fallthrough value 1, and lanes 6..7 to outer value 3.
        expected = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0]
        self.assertEqual(device.global_memory.read_floats(0, 8), expected)

    def test_divergence_is_deterministic(self):
        cycles = set()
        for _ in range(3):
            device = create_device()
            launch(
                device,
                parse_asm(NESTED_DIVERGENCE_ASM),
                grid_dim=(1,),
                block_dim=(8,),
            )
            cycles.add(device.clock)

        self.assertEqual(len(cycles), 1)


if __name__ == "__main__":
    unittest.main()
