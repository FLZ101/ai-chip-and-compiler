import unittest

from hw.isa import Imm, Mem, Reg, parse_asm, format_program


class TestIsaParser(unittest.TestCase):
    def test_parse_vector_add_program(self):
        program = parse_asm("""
            MUL R0, tid, 4
            LDG R1, [R0]
            LDG R2, [R0 + 4096]
            ADD R3, R1, R2
            STG [R0 + 8192], R3
            EXIT
            """)

        self.assertEqual(len(program), 6)
        self.assertEqual(program[0].op, "MUL")
        self.assertEqual(program[0].operands, (Reg("R0"), Reg("tid"), Imm(4)))
        self.assertEqual(program[1].operands, (Reg("R1"), Mem("R0", 0)))
        self.assertEqual(program[2].operands, (Reg("R2"), Mem("R0", 4096)))
        self.assertEqual(program[4].operands, (Mem("R0", 8192), Reg("R3")))

    def test_parse_special_math_instructions(self):
        program = parse_asm("""
            RCP R1, R2
            RSQRT R3, R4
            EXP2 R5, R6
            LOG2 R7, R8
            FMA R9, R1, R2, R3
            """)
        self.assertEqual(
            [instr.op for instr in program], ["RCP", "RSQRT", "EXP2", "LOG2", "FMA"]
        )

    def test_round_trip(self):
        asm = """
        MOV R0, tid
        MUL R1, R0, 4
        LDG R2, [R1 + 8]
        ADD R3, R2, 1
        STG [R1 + 16], R3
        EXIT
        """
        first = parse_asm(asm)
        second = parse_asm(format_program(first))
        self.assertEqual(first.instructions, second.instructions)
        self.assertEqual(first.labels, second.labels)

    def test_rejects_non_register_destination(self):
        with self.assertRaises(ValueError):
            parse_asm("ADD 3, R1, R2")

    def test_rejects_duplicate_label(self):
        with self.assertRaises(ValueError):
            parse_asm("""
                LOOP:
                ADD R0, R0, 1
                LOOP:
                EXIT
                """)


if __name__ == "__main__":
    unittest.main()
