"""Deterministic memory planning for model weights, activations, and KV cache."""

from __future__ import annotations

from dataclasses import dataclass, field

_DTYPE_WIDTHS = {
    "f32": 4,
    "f64": 8,
    "f16": 2,
    "bf16": 2,
    "i32": 4,
    "i64": 8,
    "i1": 1,
    "i8": 1,
    "u8": 1,
    "i4": 0.5,
    "u4": 0.5,
    "f8_e4m3": 1,
    "f8_e5m2": 1,
}


@dataclass(frozen=True)
class PlannedBuffer:
    """One tensor with a known live interval."""

    name: str
    numel: int
    dtype: str = "f32"
    live_from: int = 0
    live_until: int | None = None

    @property
    def byte_size(self) -> int:
        width = _DTYPE_WIDTHS[self.dtype]
        if width == 0.5:
            return (self.numel + 1) // 2
        return self.numel * int(width)


@dataclass(frozen=True)
class PlannedAllocation:
    """An assigned byte range."""

    name: str
    offset: int
    size: int
    live_until: int | None = None


@dataclass
class MemoryPlan:
    """Assign non-overlapping intervals to a shared byte region."""

    allocations: dict[str, PlannedAllocation] = field(default_factory=dict)
    peak_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        """Total bytes reserved across all planned buffers."""
        return sum(allocation.size for allocation in self.allocations.values())

    def offset(self, name: str) -> int:
        return self.allocations[name].offset

    def size(self, name: str) -> int:
        return self.allocations[name].size


def plan_memory(
    buffers: list[PlannedBuffer],
    *,
    alignment: int = 4,
) -> MemoryPlan:
    """Assign buffers with a simple first-fit allocator.

    Buffers are processed in their input order.  A buffer whose live interval
    starts after an earlier buffer's end may reuse the earlier allocation.
    """
    if alignment <= 0:
        raise ValueError("alignment must be positive")

    free: list[tuple[int, int]] = []
    allocations: dict[str, PlannedAllocation] = {}
    peak = 0

    for buffer in buffers:
        size = buffer.byte_size
        if size <= 0:
            raise ValueError(f"buffer {buffer.name!r} has non-positive size")

        free = _release_dead_buffers(
            free,
            allocations,
            buffer.live_from,
        )

        chosen = _take_free_interval(free, size, alignment)
        if chosen is None:
            cursor = peak
            cursor = _align_up(cursor, alignment)
            chosen = (cursor, cursor + size)
            peak = max(peak, chosen[1])

        allocations[buffer.name] = PlannedAllocation(
            name=buffer.name,
            offset=chosen[0],
            size=size,
            live_until=buffer.live_until,
        )

    return MemoryPlan(allocations=allocations, peak_bytes=peak)


def _release_dead_buffers(
    free: list[tuple[int, int]],
    allocations: dict[str, PlannedAllocation],
    live_from: int,
) -> list[tuple[int, int]]:
    for allocation in allocations.values():
        if allocation.live_until is not None and allocation.live_until < live_from:
            free.append((allocation.offset, allocation.offset + allocation.size))
    return _merge_intervals(sorted(free))


def _take_free_interval(
    free: list[tuple[int, int]],
    size: int,
    alignment: int,
) -> tuple[int, int] | None:
    for index, (start, end) in enumerate(free):
        offset = _align_up(start, alignment)
        if offset + size <= end:
            del free[index]
            if offset > start:
                free.append((start, offset))
            if offset + size < end:
                free.append((offset + size, end))
            free = _merge_intervals(sorted(free))
            return offset, offset + size
    return None


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment
