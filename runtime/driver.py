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
        self._queue: list[tuple[Program, tuple[int, ...], tuple[int, ...], bool]] = []

    @property
    def clock(self) -> int:
        return self.device.clock

    def allocate(
        self,
        shape: tuple[int, ...],
        *,
        name: str | None = None,
        alignment: int = 4096,
        dtype: str = "f32",
    ) -> Buffer:
        if self._cursor > 0:
            self._cursor = ((self._cursor + alignment - 1) // alignment) * alignment

        if name is None:
            name = f"buffer_{self._next_buffer_id}"
            self._next_buffer_id += 1
        if name in self._buffers:
            raise DriverError(f"buffer {name!r} already exists")

        shape_tuple = tuple(shape)
        buffer = Buffer(
            self.device,
            name=name,
            base_addr=self._cursor,
            shape=shape_tuple,
            dtype=dtype,
        )
        if any(dim < 0 for dim in shape_tuple):
            raise DriverError(f"invalid buffer shape: {shape_tuple!r}")
        if self._cursor + buffer.nbytes > len(self.device.global_memory.data):
            raise DriverError("out of simulated global memory")
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
        self.enqueue(program, grid_dim=grid_dim, block_dim=block_dim, profile=profile)
        return self.flush()

    def enqueue(
        self,
        program: Program | str,
        *,
        grid_dim: tuple[int, ...] = (1,),
        block_dim: tuple[int, ...] = (1,),
        profile: bool = False,
    ) -> None:
        """Queue one kernel for later execution by :meth:`flush`."""
        parsed = program if isinstance(program, Program) else parse_asm(program)
        self._queue.append((parsed, tuple(grid_dim), tuple(block_dim), profile))

    def flush(self) -> ProfileCounters | None:
        """Execute all queued kernels in FIFO order."""
        last_snapshot = None
        while self._queue:
            program, grid_dim, block_dim, profile = self._queue.pop(0)
            last_snapshot = launch(
                self.device,
                program,
                grid_dim=grid_dim,
                block_dim=block_dim,
                profile=profile,
            )
            self.profiler.add(
                LaunchRecord(cycles=self.device.clock, profile=last_snapshot)
            )
        return last_snapshot

    def synchronize(self) -> None:
        """Block until all previously queued work has completed."""
        self.flush()

    def sync(self) -> None:
        self.synchronize()

    def _get_buffer(self, buffer: Buffer | str) -> Buffer:
        if isinstance(buffer, Buffer):
            return buffer
        if buffer not in self._buffers:
            raise BufferError(f"unknown buffer: {buffer!r}")
        return self._buffers[buffer]
