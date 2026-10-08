"""Cycle-approximate timing model and profile counters.

Step 004 replaces the original ``one instruction per warp per cycle`` loop
with a small stall-aware scheduler.  The model is intentionally simple:

* every instruction has a fixed issue latency,
* a warp cannot issue again until that latency has elapsed,
* reads wait on a per-lane register scoreboard,
* global/shared memory share a per-cycle byte budget,
* shared-memory bank conflicts are reported as an event count.

The model is deterministic so identical launches produce identical counters.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TimingModel:
    """Configurable latencies and bandwidth limits.

    The default values are chosen to make the difference between an
    ALU-bound kernel and a memory-bound kernel obvious without making tiny
    operator tests slow.
    """

    alu_latency: int = 4
    special_latency: int = 8
    control_latency: int = 1
    barrier_latency: int = 2
    global_memory_latency: int = 20
    shared_memory_latency: int = 4
    # One full 32-lane warp access is 32 lanes * 4 bytes = 128 bytes.  Keeping
    # the per-cycle budget at or above that value lets a warp issue one memory
    # instruction as a single transaction while still leaving room to model
    # contention across multiple resident warps later.
    global_bandwidth_bytes_per_cycle: int = 128
    shared_bandwidth_bytes_per_cycle: int = 128
    bank_count: int = 32
    shared_swizzle_bits: int = 3


DEFAULT_TIMING_MODEL = TimingModel()


# Opcodes that use the slower special-function unit.  ``DIV`` is included here
# because softmax/normalization-style division is not a single-cycle ALU op on
# a minimal GPU datapath.
SPECIAL_ALU_OPS = {"DIV", "RCP", "RSQRT", "EXP2", "LOG2"}
GLOBAL_MEMORY_OPS = {"LDG", "STG"}
SHARED_MEMORY_OPS = {"LDS", "STS"}


def opcode_class(op: str) -> str:
    """Return the timing class for an opcode.

    The possible values are ``global_memory``, ``shared_memory``,
    ``special``, ``control``, and ``alu``.
    """
    if op.startswith(("LDG", "STG")):
        return "global_memory"
    if op.startswith(("LDS", "STS")):
        return "shared_memory"
    if op in SPECIAL_ALU_OPS:
        return "special"
    if op in {"BRA", "RET", "EXIT", "BAR.SYNC"}:
        return "control"
    return "alu"


def instruction_latency(op: str, model: TimingModel = DEFAULT_TIMING_MODEL) -> int:
    """Return the issue latency, in cycles, for one instruction."""
    opcode_type = opcode_class(op)
    if opcode_type == "global_memory":
        return model.global_memory_latency
    if opcode_type == "shared_memory":
        return model.shared_memory_latency
    if opcode_type == "special":
        return model.special_latency
    if opcode_type == "control":
        if op == "BAR.SYNC":
            return model.barrier_latency
        return model.control_latency
    return model.alu_latency


@dataclass
class StallCounters:
    """Per-reason stall-cycle counters for a device run."""

    fetch: int = 0
    alu: int = 0
    memory: int = 0
    barrier: int = 0
    scoreboard: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "fetch": self.fetch,
            "alu": self.alu,
            "memory": self.memory,
            "barrier": self.barrier,
            "scoreboard": self.scoreboard,
        }


@dataclass
class ProfileCounters:
    """Aggregate performance counters collected during :meth:`Device.run`."""

    cycles: int = 0
    issued_instructions: int = 0
    stalls: StallCounters = field(default_factory=StallCounters)
    global_load_bytes: int = 0
    global_store_bytes: int = 0
    shared_load_bytes: int = 0
    shared_store_bytes: int = 0
    shared_bank_conflicts: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "cycles": self.cycles,
            "issued_instructions": self.issued_instructions,
            "stalls": self.stalls.as_dict(),
            "global_load_bytes": self.global_load_bytes,
            "global_store_bytes": self.global_store_bytes,
            "shared_load_bytes": self.shared_load_bytes,
            "shared_store_bytes": self.shared_store_bytes,
            "shared_bank_conflicts": self.shared_bank_conflicts,
        }

    def snapshot(self) -> "ProfileCounters":
        """Return a deep copy safe to keep after future launches mutate state."""
        return copy.deepcopy(self)


def memory_bandwidth(op: str, model: TimingModel = DEFAULT_TIMING_MODEL) -> int:
    """Return bytes-per-cycle available to an opcode's memory space."""
    if op.startswith(("LDG", "STG")):
        return model.global_bandwidth_bytes_per_cycle
    if op.startswith(("LDS", "STS")):
        return model.shared_bandwidth_bytes_per_cycle
    raise ValueError(f"{op} is not a memory instruction")
