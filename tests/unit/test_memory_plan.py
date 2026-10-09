import unittest

from runtime.memory_plan import MemoryPlan, PlannedBuffer, plan_memory


class TestMemoryPlan(unittest.TestCase):
    def test_non_overlapping_buffers_share_addresses(self):
        plan = plan_memory(
            [
                PlannedBuffer("a", 8, "f32", live_from=0, live_until=1),
                PlannedBuffer("b", 8, "f32", live_from=2, live_until=3),
            ],
            alignment=4,
        )

        self.assertEqual(plan.offset("a"), plan.offset("b"))
        self.assertEqual(plan.peak_bytes, 32)

    def test_peak_accounts_for_live_overlap(self):
        plan = plan_memory(
            [
                PlannedBuffer("a", 8, "f32", live_from=0, live_until=2),
                PlannedBuffer("b", 8, "f32", live_from=0, live_until=2),
            ],
            alignment=4,
        )

        self.assertEqual(plan.peak_bytes, 64)
        self.assertNotEqual(plan.offset("a"), plan.offset("b"))

    def test_packed_i4_size(self):
        buffer = PlannedBuffer("packed", 8, "i4")
        self.assertEqual(buffer.byte_size, 4)


if __name__ == "__main__":
    unittest.main()
