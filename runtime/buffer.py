"""Device buffer wrapper for the simulated GPU runtime."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hw.device import Device


class BufferError(Exception):
    pass


class Buffer:
    """A contiguous FP32 allocation in simulated global memory."""

    def __init__(
        self,
        device: "Device",
        *,
        name: str,
        base_addr: int,
        shape: tuple[int, ...],
        dtype: str = "f32",
    ):
        if dtype != "f32":
            raise BufferError("only f32 buffers are supported")
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
        return self.numel * 4

    def write(self, array: np.ndarray) -> None:
        data = np.asarray(array, dtype=np.float32)
        if data.shape != self.shape:
            raise BufferError(
                f"buffer {self.name} expects shape {self.shape}, got {data.shape}"
            )
        self.device.global_memory.load_floats(self.base_addr, data.reshape(-1).tolist())

    def read(self) -> np.ndarray:
        values = self.device.global_memory.read_floats(self.base_addr, self.numel)
        return np.asarray(values, dtype=np.float32).reshape(self.shape)
