import unittest

from hw.mma import (
    DEFAULT_MMA_FRAGMENT_LAYOUT,
    FragmentLayout,
    FragmentRole,
    shared_bank_index,
)


class TestFragmentLayout(unittest.TestCase):
    def test_square_fragment_layout_covers_every_element_once(self):
        layout = DEFAULT_MMA_FRAGMENT_LAYOUT

        for role in (FragmentRole.A, FragmentRole.B, FragmentRole.C):
            seen: set[tuple[int, int]] = set()
            rows, cols = layout.matrix_shape(role)

            for row in range(rows):
                for col in range(cols):
                    slot = (
                        layout.lane_id(role, row, col),
                        layout.register_id(role, row, col),
                    )
                    self.assertNotIn(slot, seen)
                    seen.add(slot)

            self.assertEqual(len(seen), rows * cols)
            self.assertEqual(layout.elements_per_thread(role), 8)

    def test_offsets_for_lane_are_inverse_of_element_mapping(self):
        layout = DEFAULT_MMA_FRAGMENT_LAYOUT

        for role in (FragmentRole.A, FragmentRole.B, FragmentRole.C):
            rows, cols = layout.matrix_shape(role)
            for lane_id in range(layout.lane_count):
                offsets = layout.offsets_for_lane(role, lane_id)
                self.assertEqual(len(offsets), layout.elements_per_thread(role))

                for register_id, (row, col) in enumerate(offsets):
                    self.assertEqual(layout.lane_id(role, row, col), lane_id)
                    self.assertEqual(layout.register_id(role, row, col), register_id)

                self.assertEqual(
                    {(row, col) for row, col in offsets},
                    {
                        (row, col)
                        for row in range(rows)
                        for col in range(cols)
                        if layout.lane_id(role, row, col) == lane_id
                    },
                )

    def test_non_square_layout_is_rejected(self):
        with self.assertRaises(NotImplementedError):
            FragmentLayout(tile_m=16, tile_n=8, tile_k=16)


class TestSharedMemorySwizzle(unittest.TestCase):
    def test_swizzle_pattern_is_deterministic(self):
        values = [shared_bank_index(address) for address in range(1024)]
        self.assertEqual(
            values,
            [shared_bank_index(address) for address in range(1024)],
        )

    def test_swizzle_changes_linear_bank_for_known_address(self):
        # Word 128 has linear bank 0 but is moved to bank 16 by the 128-byte
        # swizzle.  The byte address is therefore 128 * 4.
        self.assertEqual(shared_bank_index(128 * 4), 16)

        # Disabling the swizzle restores the simple linear bank mapping.
        self.assertEqual(shared_bank_index(128 * 4, swizzle_bits=0), 0)

    def test_swizzle_rejects_non_power_of_two_bank_count(self):
        with self.assertRaises(ValueError):
            shared_bank_index(0, bank_count=24)


if __name__ == "__main__":
    unittest.main()
