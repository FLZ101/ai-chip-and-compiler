import unittest

import numpy as np

from runtime import Driver


class TestRuntimeDriver(unittest.TestCase):
    """Step 015 runtime/driver tests."""

    def test_multiple_launches_are_deterministic(self):
        driver = Driver(global_size=1 << 20, sm_count=4)
        a = driver.allocate((8,), name="a")
        b = driver.allocate((8,), name="b")
        out = driver.allocate((8,), name="out")

        a_data = np.arange(8, dtype=np.float32) + 1.0
        b_data = np.arange(8, dtype=np.float32) * 0.5
        driver.copy_to_device(a, a_data)
        driver.copy_to_device(b, b_data)

        isa = f"""
MOV R4, {a.base_addr}
MUL R0, tid.x, 4
ADD R0, R0, R4
LDG R1, [R0]
LDG R2, [R0 + {b.base_addr - a.base_addr}]
ADD R3, R1, R2
STG [R0 + {out.base_addr - a.base_addr}], R3
EXIT
"""
        first = driver.launch(isa, grid_dim=(1,), block_dim=(8,), profile=True)
        first_out = driver.copy_from_device(out)

        driver.copy_to_device(a, a_data * 2.0)
        driver.copy_to_device(b, b_data)
        second = driver.launch(isa, grid_dim=(1,), block_dim=(8,), profile=True)
        second_out = driver.copy_from_device(out)

        np.testing.assert_allclose(first_out, a_data + b_data)
        np.testing.assert_allclose(second_out, a_data * 2.0 + b_data)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first.issued_instructions, second.issued_instructions)

        summary = driver.profiler.summary()
        self.assertEqual(summary["launches"], 2)
        self.assertGreater(summary["total_cycles"], 0)

    def test_unknown_buffer_rejected(self):
        driver = Driver(global_size=1 << 20, sm_count=4)
        with self.assertRaises(Exception):
            driver.copy_from_device("missing")


if __name__ == "__main__":
    unittest.main()
