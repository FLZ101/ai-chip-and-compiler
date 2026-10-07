import unittest

from hw.memory import ByteMemory, MemoryError


class TestTypedMemory(unittest.TestCase):
    def test_f16_round_trip(self):
        memory = ByteMemory(32)
        memory.write_f16(0, 1.5)
        self.assertEqual(memory.read_f16(0), 1.5)

    def test_bf16_round_trip(self):
        memory = ByteMemory(32)
        memory.write_bf16(0, 2.0)
        self.assertEqual(memory.read_bf16(0), 2.0)

    def test_i8_and_u8_round_trip(self):
        memory = ByteMemory(32)
        memory.write_i8(0, -7)
        memory.write_u8(1, 250)
        self.assertEqual(memory.read_i8(0), -7)
        self.assertEqual(memory.read_u8(1), 250)

    def test_i32_round_trip(self):
        memory = ByteMemory(32)
        memory.write_i32(0, -123456)
        self.assertEqual(memory.read_i32(0), -123456)

    def test_typed_accesses_check_bounds(self):
        memory = ByteMemory(4)
        with self.assertRaises(MemoryError):
            memory.read_f16(3)


if __name__ == "__main__":
    unittest.main()
