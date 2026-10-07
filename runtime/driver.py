"""Host-side driver for the simulated GPU."""

from __future__ import annotations

from typing import Any

import numpy as np

from hw.device import Device, create_device, launch
from hw.isa import Program, parse_asm
from hw.timing import ProfileCounters

from .buffer import Buffer, BufferError
from .profiler import LaunchRecord, RuntimeProfiler


class DriverError(Exception):
    pass


class Driver:
    """Own a simulated device and expose a stable host/device API."""

    def __init__(
        self,
        *,
        device: Device | None = None,
        global_size: int = 1 << 20,
        sm_count: int = 4,
    ):
        self.device = device or create_device(
            global_size=global_size, sm_count=sm_count
        )
        self.profiler = RuntimeProfiler()
        self._buffers: dict[str, Buffer] = {}
        self._next_buffer_id = 0
        self._cursor = 0

    @property
    def clock(self) -> int:
        return self.device.clock

    def allocate(
        self,
        shape: tuple[int, ...],
        *,
        name: str | None = None,
        alignment: int = 4096,
    ) -> Buffer:
        if self._cursor > 0:
            self._cursor = ((self._cursor + alignment - 1) // alignment) * alignment

        if name is None:
            name = f"buffer_{self._next_buffer_id}"
            self._next_buffer_id += 1
        if name in self._buffers:
            raise DriverError(f"buffer {name!r} already exists")

        shape_tuple = tuple(shape)
        numel = 1
        for dim in shape_tuple:
            if dim < 0:
                raise DriverError(f"invalid buffer shape: {shape_tuple!r}")
            numel *= dim
        if self._cursor + numel * 4 > len(self.device.global_memory.data):
            raise DriverError("out of simulated global memory")

        buffer = Buffer(
            self.device,
            name=name,
            base_addr=self._cursor,
            shape=shape_tuple,
        )
        self._buffers[name] = buffer
        self._cursor = buffer.base_addr + buffer.nbytes

        buffer.write(np.zeros(buffer.shape, dtype=np.float32))
        return buffer

    def copy_to_device(self, buffer: Buffer | str, array: np.ndarray) -> None:
        self._get_buffer(buffer).write(array)

    def copy_from_device(self, buffer: Buffer | str) -> np.ndarray:
        return self._get_buffer(buffer).read()

    def launch(
        self,
        program: Program | str,
        *,
        grid_dim: tuple[int, ...] = (1,),
        block_dim: tuple[int, ...] = (1,),
        profile: bool = False,
    ) -> ProfileCounters | None:
        parsed = program if isinstance(program, Program) else parse_asm(program)
        snapshot = launch(
            self.device,
            parsed,
            grid_dim=grid_dim,
            block_dim=block_dim,
            profile=profile,
        )
        self.profiler.add(LaunchRecord(cycles=self.device.clock, profile=snapshot))
        return snapshot

    def sync(self) -> None:
        # The current simulator executes synchronously; this method preserves
        # the host API for later asynchronous execution.
        return None

    def _get_buffer(self, buffer: Buffer | str) -> Buffer:
        if isinstance(buffer, Buffer):
            return buffer
        if buffer not in self._buffers:
            raise BufferError(f"unknown buffer: {buffer!r}")
        return self._buffers[buffer]
