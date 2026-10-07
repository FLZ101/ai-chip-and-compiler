"""Memory hierarchy for the minimal GPU simulator."""

from __future__ import annotations

import struct


class MemoryError(Exception):
    pass


class ByteMemory:
    """A byte-addressable memory region with bounds checking.

    Both global and shared memory use this class.  Loads and stores in the MVP
    are 32-bit FP32 accesses.  The simulator stores values in a raw
    ``bytearray`` and converts with little-endian IEEE-754 via ``struct``.
    """

    def __init__(self, size: int, *, name: str = "memory"):
        if size < 0:
            raise ValueError("memory size must be non-negative")
        self.name = name
        self.data = bytearray(size)

    def check(self, addr: int, count: int) -> None:
        # Keep every access inside the backing array.  This catches invalid
        # address calculations early instead of allowing silent corruption.
        if addr < 0 or count < 0 or addr + count > len(self.data):
            raise MemoryError(
                f"{self.name} access out of bounds: addr={addr}, bytes={count}, size={len(self.data)}"
            )

    def read_f32(self, addr: int) -> float:
        # GPU global/shared loads are naturally aligned in the MVP.  A 4-byte
        # read at an arbitrary address is still supported because the memory
        # model is byte-addressable.
        self.check(addr, 4)
        return struct.unpack_from("<f", self.data, addr)[0]

    def write_f32(self, addr: int, value: float) -> None:
        self.check(addr, 4)
        struct.pack_into("<f", self.data, addr, value)

    def read_u32(self, addr: int) -> int:
        self.check(addr, 4)
        return struct.unpack_from("<I", self.data, addr)[0]

    def write_u32(self, addr: int, value: int) -> None:
        self.check(addr, 4)
        struct.pack_into("<I", self.data, addr, value)

    def load_floats(self, base_addr: int, values: list[float]) -> None:
        # Convenience helper used by tests and, later, by the runtime driver to
        # copy host tensors into device memory.
        for i, value in enumerate(values):
            self.write_f32(base_addr + 4 * i, value)

    def read_floats(self, base_addr: int, count: int) -> list[float]:
        return [self.read_f32(base_addr + 4 * i) for i in range(count)]


class RegisterFile:
    """Per-thread virtual register file.

    A real GPU register file has a fixed width and hardware hazard tracking.
    This class models the programmer-visible state; the per-lane scoreboard is
    implemented by the execution core in ``hw/core.py``.
    """

    def __init__(self, count: int = 64):
        self.regs = {f"R{i}": 0 for i in range(count)}

    def read(self, name: str) -> int | float:
        if name not in self.regs:
            raise MemoryError(f"unknown register: {name}")
        return self.regs[name]

    def write(self, name: str, value: int | float) -> None:
        if name not in self.regs:
            raise MemoryError(f"unknown register: {name}")
        self.regs[name] = value


def as_int(value: int | float) -> int:
    """Convert an FP32 register value to a valid integer address.

    This is intentionally strict: non-integral addresses are rejected rather
    than silently truncated, because they almost always indicate a codegen bug.
    """
    if isinstance(value, int):
        return value
    if float(value).is_integer():
        return int(value)
    raise MemoryError(f"expected integer address value, got {value!r}")
