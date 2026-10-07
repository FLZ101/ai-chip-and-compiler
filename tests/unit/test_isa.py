import unittest

from hw.isa import Imm, Label, Mem, Reg, parse_asm, format_program


class TestIsaParser(unittest.TestCase):
    """Tests for ISA assembly parsing and formatting."""

    def test_parse_vector_add_program(self):
        # This is the first real ISA program.  It multiplies the thread id by 4
        # bytes, loads two floats, adds them, stores the result, and exits.
        program = parse_asm("""
            MUL R0, tid.x, 4
            LDG R1, [R0]
            LDG R2, [R0 + 4096]
            ADD R3, R1, R2
            STG [R0 + 8192], R3
            EXIT
            """)

        self.assertEqual(len(program), 6)
        self.assertEqual(program[0].op, "MUL")
        self.assertEqual(program[0].operands, (Reg("R0"), Reg("tid.x"), Imm(4)))
        self.assertEqual(program[1].operands, (Reg("R1"), Mem("R0", 0)))
        self.assertEqual(program[2].operands, (Reg("R2"), Mem("R0", 4096)))
        self.assertEqual(program[4].operands, (Mem("R0", 8192), Reg("R3")))

    def test_parse_special_math_instructions(self):
        # Transformer kernels need reciprocal, reciprocal-square-root, exp2,
        # log2, and fused multiply-add for softmax and normalization.
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

    def test_parse_integer_division(self):
        program = parse_asm("IDIV R1, R2, R3")
        self.assertEqual(program[0].op, "IDIV")

    def test_parse_typed_memory_accesses(self):
        program = parse_asm("""
            LDG.F16 R0, [R1]
            STG.BF16 [R1], R0
            LDS.I8 R2, [R3]
            STS.U8 [R3], R2
            LDG.I32 R4, [R5]
            STG.I32 [R5], R4
            LDG.I4 R6, [R7]
            STG.U4 [R7], R6
        """)

        self.assertEqual(
            [instr.op for instr in program],
            [
                "LDG.F16",
                "STG.BF16",
                "LDS.I8",
                "STS.U8",
                "LDG.I32",
                "STG.I32",
                "LDG.I4",
                "STG.U4",
            ],
        )

    def test_round_trip(self):
        # Round-tripping through the formatter must produce the same program.
        # This guarantees the MLIR backend can emit and reload ISA text later.
        asm = """
        MOV R0, tid.x
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
        # An immediate cannot be written by an ALU instruction.
        with self.assertRaises(ValueError):
            parse_asm("ADD 3, R1, R2")

    def test_rejects_duplicate_label(self):
        # Labels are the future branch target namespace, so duplicates must be
        # rejected early instead of producing ambiguous branch addresses.
        with self.assertRaises(ValueError):
            parse_asm("""
                LOOP:
                ADD R0, R0, 1
                LOOP:
                EXIT
                """)

    def test_parse_conditional_branch_and_compare(self):
        program = parse_asm("""
            CMP.LT R0, tid.x, 4
            BRA else, R0, end
            MOV R1, 1
            BRA end
            else:
            MOV R1, 2
            BRA end
            end:
            EXIT
            """)

        self.assertEqual(program[0].op, "CMP.LT")
        self.assertEqual(program[1].op, "BRA")
        self.assertEqual(
            program[1].operands,
            (Label("else"), Reg("R0"), Label("end")),
        )

    def test_rejects_undefined_branch_label(self):
        with self.assertRaises(ValueError):
            parse_asm("BRA missing")

    def test_parse_3d_special_registers(self):
        program = parse_asm("""
            MOV R0, ctaid.x
            MOV R1, ctaid.y
            MOV R2, ctaid.z
            MOV R3, tid.x
            MOV R4, nctaid.x
            MOV R5, ntid.x
            """)

        self.assertEqual(
            [instr.operands[1].name for instr in program],
            ["ctaid.x", "ctaid.y", "ctaid.z", "tid.x", "nctaid.x", "ntid.x"],
        )


if __name__ == "__main__":
    unittest.main()
