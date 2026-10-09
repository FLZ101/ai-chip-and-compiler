"""Device buffer wrapper for the simulated GPU runtime."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hw.device import Device


_DTYPE_WIDTHS = {
    "f32": 4,
    "f16": 2,
    "bf16": 2,
    "i32": 4,
    "i8": 1,
    "u8": 1,
    "i4": 0.5,
    "u4": 0.5,
    "f8_e4m3": 1,
    "f8_e5m2": 1,
}


class BufferError(Exception):
    pass


class Buffer:
    """A contiguous typed allocation in simulated global memory."""

    def __init__(
        self,
        device: "Device",
        *,
        name: str,
        base_addr: int,
        shape: tuple[int, ...],
        dtype: str = "f32",
    ):
        if dtype not in _DTYPE_WIDTHS:
            raise BufferError(f"unsupported buffer dtype: {dtype!r}")
        if any(dim < 0 for dim in shape):
            raise BufferError(f"invalid buffer shape: {shape!r}")
        self.device = device
        self.name = name
        self.base_addr = base_addr
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype = dtype

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        width = _DTYPE_WIDTHS[self.dtype]
        if width == 0.5:
            return (self.numel + 1) // 2
        return self.numel * int(width)

    def write(self, array: np.ndarray) -> None:
        if tuple(array.shape) != self.shape:
            raise BufferError(
                f"buffer {self.name} expects shape {self.shape}, got {tuple(array.shape)}"
            )
        values = np.asarray(array).reshape(-1)
        memory = self.device.global_memory
        write = _WRITE_METHODS[self.dtype]
        for index, value in enumerate(values.tolist()):
            write(memory, self._element_addr(index), value)

    def read(self) -> np.ndarray:
        memory = self.device.global_memory
        read = _READ_METHODS[self.dtype]
        values = [
            read(memory, self._element_addr(index)) for index in range(self.numel)
        ]
        return np.asarray(values).reshape(self.shape)

    def _element_addr(self, index: int) -> int:
        if self.dtype in {"i4", "u4"}:
            return self.base_addr * 2 + index
        width = int(_DTYPE_WIDTHS[self.dtype])
        return self.base_addr + index * width


def _write_f16(memory, addr, value):
    memory.write_f16(addr, value)


def _read_f16(memory, addr):
    return memory.read_f16(addr)


def _write_bf16(memory, addr, value):
    memory.write_bf16(addr, value)


def _read_bf16(memory, addr):
    return memory.read_bf16(addr)


def _write_i8(memory, addr, value):
    memory.write_i8(addr, int(value))


def _read_i8(memory, addr):
    return memory.read_i8(addr)


def _write_u8(memory, addr, value):
    memory.write_u8(addr, int(value))


def _read_u8(memory, addr):
    return memory.read_u8(addr)


def _write_i4(memory, addr, value):
    memory.write_i4(addr, int(value))


def _read_i4(memory, addr):
    return memory.read_i4(addr)


def _write_u4(memory, addr, value):
    memory.write_u4(addr, int(value))


def _read_u4(memory, addr):
    return memory.read_u4(addr)


def _write_i32(memory, addr, value):
    memory.write_i32(addr, int(value))


def _read_i32(memory, addr):
    return memory.read_i32(addr)


def _write_f8_e4m3(memory, addr, value):
    memory.write_f8_e4m3(addr, value)


def _read_f8_e4m3(memory, addr):
    return memory.read_f8_e4m3(addr)


def _write_f8_e5m2(memory, addr, value):
    memory.write_f8_e5m2(addr, value)


def _read_f8_e5m2(memory, addr):
    return memory.read_f8_e5m2(addr)


_WRITE_METHODS = {
    "f32": lambda memory, addr, value: memory.write_f32(addr, value),
    "f16": _write_f16,
    "bf16": _write_bf16,
    "i32": _write_i32,
    "i8": _write_i8,
    "u8": _write_u8,
    "i4": _write_i4,
    "u4": _write_u4,
    "f8_e4m3": _write_f8_e4m3,
    "f8_e5m2": _write_f8_e5m2,
}

_READ_METHODS = {
    "f32": lambda memory, addr: memory.read_f32(addr),
    "f16": _read_f16,
    "bf16": _read_bf16,
    "i32": _read_i32,
    "i8": _read_i8,
    "u8": _read_u8,
    "i4": _read_i4,
    "u4": _read_u4,
    "f8_e4m3": _read_f8_e4m3,
    "f8_e5m2": _read_f8_e5m2,
}
