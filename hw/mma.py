"""MMA fragment layout and shared-memory swizzle helpers.

Step 4.2 adds the two hardware concepts needed before the compiler can lower
tiled matmuls to tensor-core ``MMA`` instructions:

* A/B/C fragment layouts describe how the elements of one MMA tile are spread
  across the 32 lanes of a warp.
* The shared-memory swizzle pattern describes how a word address is mapped to
  a physical shared-memory bank.  The simulator uses this mapping only for
  bank-conflict accounting, while functional loads and stores continue to use
  the logical byte address.

The MVP targets a square ``16x16x16`` tensor-core tile.  Larger and
non-square tiles can be added later by generalizing
:class:`FragmentLayout`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FragmentRole(str, Enum):
    """Role of a fragment in an MMA instruction."""

    A = "A"
    B = "B"
    C = "C"


@dataclass(frozen=True)
class FragmentLayout:
    """Square 16x16x16 fragment layout for a 32-lane warp.

    The A and B operands are ``16x16`` matrices and the C/D accumulator is also
    a ``16x16`` matrix.  The simulator uses a simple 8x4 lane tile:

    * ``lane = (row % 8) * 4 + (col % 4)``
    * ``register = (row // 8) * 4 + (col // 4)``

    This gives every lane exactly eight register slots for every role.  The
    layout is deterministic and covers each logical element exactly once.
    """

    tile_m: int = 16
    tile_n: int = 16
    tile_k: int = 16
    lane_count: int = 32

    def __post_init__(self) -> None:
        # Keep the first fragment implementation small and easy to reason
        # about.  The compiler pass in step 4.3 will emit only square 16x16x16
        # tiles; non-square layouts can be introduced when needed.
        if self.tile_m != self.tile_n or self.tile_m != self.tile_k:
            raise NotImplementedError("only square MMA tiles are supported")
        if self.tile_m != 16 or self.tile_n != 16 or self.tile_k != 16:
            raise NotImplementedError("only 16x16x16 MMA tiles are supported")
        if self.lane_count != 32:
            raise NotImplementedError("fragment layout requires a 32-lane warp")

    def matrix_shape(self, role: FragmentRole) -> tuple[int, int]:
        """Return the logical ``(rows, cols)`` shape for *role*."""
        if role == FragmentRole.A:
            return (self.tile_m, self.tile_k)
        if role == FragmentRole.B:
            return (self.tile_k, self.tile_n)
        if role == FragmentRole.C:
            return (self.tile_m, self.tile_n)
        raise ValueError(f"unknown fragment role: {role!r}")

    def elements_per_thread(self, role: FragmentRole) -> int:
        """Return the number of register slots each lane owns for *role*."""
        rows, cols = self.matrix_shape(role)
        elements = rows * cols
        if elements % self.lane_count != 0:
            raise ValueError("fragment matrix does not divide evenly across lanes")
        return elements // self.lane_count

    def lane_id(self, role: FragmentRole, row: int, col: int) -> int:
        """Return the lane that owns logical element ``(row, col)``."""
        rows, cols = self.matrix_shape(role)
        self._validate_coordinate(rows, cols, row, col)
        return (row % 8) * 4 + (col % 4)

    def register_id(self, role: FragmentRole, row: int, col: int) -> int:
        """Return the per-lane register slot for ``(row, col)``."""
        rows, cols = self.matrix_shape(role)
        self._validate_coordinate(rows, cols, row, col)
        return (row // 8) * 4 + (col // 4)

    def offsets_for_lane(
        self, role: FragmentRole, lane_id: int
    ) -> list[tuple[int, int]]:
        """Return ``(row, col)`` coordinates owned by *lane_id*.

        The order is deterministic: row groups are iterated before column
        groups, and within a group rows and columns are iterated in ascending
        order.  This makes the inverse mapping in unit tests exact.
        """
        if not 0 <= lane_id < self.lane_count:
            raise ValueError(f"lane_id out of range: {lane_id}")

        rows, cols = self.matrix_shape(role)
        lane_row = lane_id // 4
        lane_col = lane_id % 4
        row_groups = rows // 8
        col_groups = cols // 4
        return [
            (lane_row + row_group * 8, lane_col + col_group * 4)
            for row_group in range(row_groups)
            for col_group in range(col_groups)
        ]

    @staticmethod
    def _validate_coordinate(rows: int, cols: int, row: int, col: int) -> None:
        if not 0 <= row < rows:
            raise ValueError(f"row out of range: {row}")
        if not 0 <= col < cols:
            raise ValueError(f"column out of range: {col}")


DEFAULT_MMA_FRAGMENT_LAYOUT = FragmentLayout()


def shared_bank_index(
    address: int, *, bank_count: int = 32, swizzle_bits: int = 3
) -> int:
    """Map a shared-memory byte address to a physical bank index.

    The default pattern is the familiar 128-byte PTX-style swizzle:

    ``word = address // 4``
    ``swizzled_word = word ^ (((word >> 7) & 0x7) << 4)``
    ``bank = swizzled_word & 31``

    This permutation does not change the functional contents of shared memory;
    it changes which lanes are considered to access the same physical bank.
    """
    if bank_count <= 0 or bank_count & (bank_count - 1):
        raise ValueError("bank_count must be a positive power of two")
    if swizzle_bits < 0:
        raise ValueError("swizzle_bits must be non-negative")

    word = address // 4
    bank_bits = bank_count.bit_length() - 1
    group_shift = bank_bits + swizzle_bits - 1
    xor_shift = bank_bits - 1

    if swizzle_bits == 0:
        return word & (bank_count - 1)

    swizzle_mask = (1 << swizzle_bits) - 1
    swizzled_word = word ^ (((word >> group_shift) & swizzle_mask) << xor_shift)
    return swizzled_word & (bank_count - 1)
