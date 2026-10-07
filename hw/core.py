"""Minimal SIMT execution core."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .isa import (
    MEMORY_DEST_OPS,
    REGISTER_DEST_OPS,
    Imm,
    Instruction,
    Label,
    Mem,
    Program,
    Reg,
)
from .memory import ByteMemory, RegisterFile, as_int
from .timing import (
    DEFAULT_TIMING_MODEL,
    ProfileCounters,
    TimingModel,
    instruction_latency,
    memory_bandwidth,
    opcode_class,
)

SPECIAL_REGISTER_NAMES = {
    "tid.x",
    "tid.y",
    "tid.z",
    "ntid.x",
    "ntid.y",
    "ntid.z",
    "ctaid.x",
    "ctaid.y",
    "ctaid.z",
    "nctaid.x",
    "nctaid.y",
    "nctaid.z",
    "laneid",
    "warpid",
    "clock",
}


@dataclass
class LaneState:
    """State for one SIMT lane.

    Lanes in the same warp execute the same instruction, but each has an
    independent register file and ``active`` flag for divergence support.
    """

    lane_id: int
    regs: RegisterFile = field(default_factory=RegisterFile)
    active: bool = True
    # Maps register name to the cycle when its most recent write becomes
    # visible.  A value is written into the register file at issue for
    # simplicity, but the scoreboard prevents dependent instructions from
    # reading it before this ready cycle.
    reg_ready_cycle: dict[str, int] = field(default_factory=dict)


@dataclass
class WarpState:
    """State for a 32-lane warp.

    ``active_mask`` is the SIMT execution mask.  Branch reconvergence was added
    in step 002, and ``waiting_at_barrier`` is used by block barriers in step
    003.
    """

    warp_id: int
    lanes: list[LaneState]
    pc: int = 0
    exited: bool = False
    active_mask: int = 0xFFFFFFFF
    reconvergence_stack: list["ReconvergenceState"] = field(default_factory=list)
    waiting_at_barrier: bool = False
    # Cycle after which this warp may issue its next instruction.
    busy_until: int = 0
    # Timing class of the instruction currently occupying the warp.
    last_issued_class: str = "none"

    @property
    def active_lanes(self) -> list[LaneState]:
        # A lane participates in an instruction only when both the lane and its
        # bit in the warp active mask are enabled.
        return [
            lane
            for lane in self.lanes
            if lane.active and self.active_mask & (1 << lane.lane_id)
        ]


@dataclass
class ReconvergenceState:
    """Bookkeeping for one pending side of a divergent branch.

    ``phase`` is ``"taken"`` while the original taken path is running and
    switches to ``"fallthrough"`` when the saved fallthrough path is running.
    """

    pending_pc: int
    pending_mask: int
    reconvergence_pc: int
    original_mask: int
    phase: str = "taken"


@dataclass
class SM:
    """Streaming multiprocessor.

    Each SM owns private shared memory and zero or more resident warps.
    """

    sm_id: int
    shared_memory: ByteMemory
    warps: list[WarpState] = field(default_factory=list)
    clock: int = 0
    next_warp_index: int = 0


@dataclass
class Device:
    """Top-level simulated GPU device."""

    global_memory: ByteMemory
    sms: list[SM]
    grid_dim: tuple[int, ...] = (1,)
    block_dim: tuple[int, ...] = (1,)
    block_idx: tuple[int, ...] = (0,)
    clock: int = 0
    timing_model: TimingModel = field(default_factory=lambda: DEFAULT_TIMING_MODEL)
    profile: ProfileCounters = field(default_factory=ProfileCounters)
    _cycle_global_bytes: int = field(default=0, init=False, repr=False)
    _cycle_shared_bytes: int = field(default=0, init=False, repr=False)

    def run(self, program: Program) -> None:
        # One global cycle advances the deterministic timing model.  Each SM
        # has one scheduler issue slot per cycle; a warp is chosen round-robin
        # from the warps that are ready to issue.
        self.profile = ProfileCounters()
        while True:
            self.clock += 1
            self._cycle_global_bytes = 0
            self._cycle_shared_bytes = 0
            self.profile.cycles += 1
            for sm in self.sms:
                sm.clock = self.clock
                self._step_sm(sm, program)

            if all(warp.exited for sm in self.sms for warp in sm.warps):
                break

    def _step_sm(self, sm: SM, program: Program) -> None:
        # Resolve reconvergence and end-of-program transitions before the
        # scheduler decides which warp, if any, can issue this cycle.
        for warp in sm.warps:
            if warp.exited or warp.waiting_at_barrier:
                continue
            while _apply_reconvergence(warp):
                pass
            if warp.pc >= len(program):
                warp.exited = True

        selected_index = self._select_ready_warp(sm, program)
        if selected_index is None:
            for warp in sm.warps:
                if not warp.exited:
                    self._count_warp_stall(warp, program)
            return

        selected_warp = sm.warps[selected_index]
        self._issue_instruction(sm, selected_warp, program)
        sm.next_warp_index = (selected_index + 1) % len(sm.warps)

        for index, warp in enumerate(sm.warps):
            if index == selected_index or warp.exited:
                continue
            self._count_warp_stall(warp, program)

    def _select_ready_warp(self, sm: SM, program: Program) -> int | None:
        if not sm.warps:
            return None
        order = [
            (sm.next_warp_index + offset) % len(sm.warps)
            for offset in range(len(sm.warps))
        ]
        for index in order:
            if self._warp_can_issue(sm, sm.warps[index], program):
                return index
        return None

    def _warp_can_issue(self, sm: SM, warp: WarpState, program: Program) -> bool:
        if warp.exited or warp.waiting_at_barrier:
            return False
        if warp.busy_until > self.clock:
            return False

        instruction = program[warp.pc]
        if self._has_scoreboard_stall(warp, instruction):
            return False

        if opcode_class(instruction.op) in {"global_memory", "shared_memory"}:
            bytes_needed = len(warp.active_lanes) * 4
            used = (
                self._cycle_global_bytes
                if opcode_class(instruction.op) == "global_memory"
                else self._cycle_shared_bytes
            )
            if used + bytes_needed > memory_bandwidth(
                instruction.op, self.timing_model
            ):
                return False
        return True

    def _issue_instruction(self, sm: SM, warp: WarpState, program: Program) -> None:
        instruction = program[warp.pc]
        opcode_type = opcode_class(instruction.op)
        active_lane_count = len(warp.active_lanes)

        self.profile.issued_instructions += 1

        if opcode_type == "global_memory":
            self._cycle_global_bytes += active_lane_count * 4
            if instruction.op == "LDG":
                self.profile.global_load_bytes += active_lane_count * 4
            else:
                self.profile.global_store_bytes += active_lane_count * 4
        elif opcode_type == "shared_memory":
            self._cycle_shared_bytes += active_lane_count * 4
            if instruction.op == "LDS":
                self.profile.shared_load_bytes += active_lane_count * 4
            else:
                self.profile.shared_store_bytes += active_lane_count * 4
            self.profile.shared_bank_conflicts += _detect_shared_bank_conflicts(
                self, sm, warp, instruction
            )

        # The functional side effect is applied immediately.  The timing state
        # below prevents the next instruction from observing the result until
        # the configured latency has passed.
        execute_warp(self, sm, warp, instruction, program)

        latency = instruction_latency(instruction.op, self.timing_model)
        warp.busy_until = self.clock + latency
        warp.last_issued_class = opcode_type

        if instruction.op in REGISTER_DEST_OPS:
            dst = instruction.dst
            if not isinstance(dst, Reg):
                raise ValueError(f"{instruction.op} requires a register destination")
            for lane in warp.active_lanes:
                lane.reg_ready_cycle[dst.name] = self.clock + latency

    def _has_scoreboard_stall(self, warp: WarpState, instruction: Instruction) -> bool:
        sources = _instruction_source_register_names(instruction)
        for lane in warp.active_lanes:
            for register in sources:
                if lane.reg_ready_cycle.get(register, 0) > self.clock:
                    return True
        return False

    def _count_warp_stall(self, warp: WarpState, program: Program) -> None:
        if warp.waiting_at_barrier:
            self.profile.stalls.barrier += 1
            return

        instruction = program[warp.pc]
        # A dependent instruction is blocked on its producer's writeback even
        # when the producer's issue latency also keeps the warp busy.  Counting
        # the scoreboard first makes register dependency stalls visible instead
        # of folding them all into generic ALU/memory latency.
        if self._has_scoreboard_stall(warp, instruction):
            self.profile.stalls.scoreboard += 1
            return

        if warp.busy_until > self.clock:
            if warp.last_issued_class in {"global_memory", "shared_memory"}:
                self.profile.stalls.memory += 1
            else:
                self.profile.stalls.alu += 1
            return

        if opcode_class(instruction.op) in {"global_memory", "shared_memory"}:
            bytes_needed = len(warp.active_lanes) * 4
            used = (
                self._cycle_global_bytes
                if opcode_class(instruction.op) == "global_memory"
                else self._cycle_shared_bytes
            )
            if used + bytes_needed > memory_bandwidth(
                instruction.op, self.timing_model
            ):
                self.profile.stalls.memory += 1
                return

        # The warp is ready but either lost the round-robin issue slot or is
        # otherwise waiting on the front end.  This models an SM with a single
        # instruction fetch/issue path shared by its resident warps.
        self.profile.stalls.fetch += 1


def _make_lanes(block_threads: int) -> list[LaneState]:
    return [LaneState(lane_id=i) for i in range(block_threads)]


def create_device(
    global_size: int = 1 << 20,
    shared_size: int = 1 << 16,
    sm_count: int = 4,
    *,
    timing_model: TimingModel | None = None,
) -> Device:
    # The default sizes are intentionally small so tests run quickly.  The
    # driver/runtime will eventually allocate tensors against these regions.
    return Device(
        global_memory=ByteMemory(global_size, name="global"),
        sms=[
            SM(sm_id=i, shared_memory=ByteMemory(shared_size, name=f"shared{i}"))
            for i in range(sm_count)
        ],
        timing_model=timing_model or DEFAULT_TIMING_MODEL,
    )


def launch(
    device: Device,
    program: Program,
    *,
    grid_dim: tuple[int, ...] = (1,),
    block_dim: tuple[int, ...] = (1,),
    profile: bool = False,
) -> ProfileCounters | None:
    # Launch dimensions are runtime values in a real GPU, but in the MVP they
    # are function arguments.  Keeping them as tuples makes the eventual
    # 1D/2D/3D grid handling straightforward.
    device.grid_dim = grid_dim
    device.block_dim = block_dim
    block_threads = math.prod(block_dim)
    total_blocks = math.prod(grid_dim)
    if total_blocks > len(device.sms):
        # A real GPU schedules blocks over time.  The MVP assigns at most one
        # block per SM, which keeps control flow and memory visibility simple.
        raise ValueError(
            f"grid has {total_blocks} blocks but device only has {len(device.sms)} SMs"
        )

    for block_linear_id in range(total_blocks):
        sm = device.sms[block_linear_id]
        sm.warps = []
        # A real GPU groups the threads of each block into 32-lane warps.  The
        # final warp may be partially populated, and its active mask covers only
        # the lanes that exist.
        for warp_index in range(math.ceil(block_threads / 32)):
            lane_count = min(32, block_threads - warp_index * 32)
            lanes = _make_lanes(lane_count)
            active_mask = (1 << lane_count) - 1
            warp = WarpState(
                warp_id=warp_index,
                lanes=lanes,
                active_mask=active_mask,
            )
            sm.warps.append(warp)
        # Record the block this SM is executing.  Special register evaluation
        # uses these values to derive global thread IDs.
        sm._block_idx = _unflatten_index(block_linear_id, grid_dim)  # type: ignore[attr-defined]
        sm._block_linear_id = block_linear_id  # type: ignore[attr-defined]

    device.run(program)
    if profile:
        return device.profile.snapshot()
    return None


def _unflatten_index(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    # Convert a linear block index to a multi-dimensional CUDA-style index.
    # For example, shape (3, 4) maps linear index 6 to (1, 2).
    result: list[int] = []
    for dim in reversed(shape):
        result.append(index % dim)
        index //= dim
    return tuple(reversed(result))


def _special_register_value(
    device: Device, sm: SM, warp: WarpState, lane: LaneState, name: str
) -> int:
    # Special registers are evaluated on demand.  This keeps the execution
    # loop agnostic to how the hardware scheduler derives thread/block IDs.
    if name == "laneid":
        return lane.lane_id
    if name == "warpid":
        return warp.warp_id
    if name in {"tid.x", "tid.y", "tid.z"}:
        block_linear_id = getattr(sm, "_block_linear_id", 0)
        block_threads = math.prod(device.block_dim)
        global_thread_id = (
            block_linear_id * block_threads + warp.warp_id * 32 + lane.lane_id
        )
        block_local_thread_id = global_thread_id % block_threads
        thread_idx = _unflatten_index(block_local_thread_id, device.block_dim)
        return _component_or_default(thread_idx, _dimension_index(name), 0)
    if name in {"ntid.x", "ntid.y", "ntid.z"}:
        return _component_or_default(device.block_dim, _dimension_index(name), 1)
    if name in {"ctaid.x", "ctaid.y", "ctaid.z"}:
        block_idx = getattr(sm, "_block_idx", (0,))
        return _component_or_default(block_idx, _dimension_index(name), 0)
    if name in {"nctaid.x", "nctaid.y", "nctaid.z"}:
        return _component_or_default(device.grid_dim, _dimension_index(name), 1)
    if name == "clock":
        return device.clock
    raise ValueError(f"unknown special register: {name}")


def _component_or_default(values: tuple[int, ...], dimension: int, default: int) -> int:
    """Return ``values[dimension]`` or a default for omitted dimensions."""
    if dimension < len(values):
        return values[dimension]
    return default


def _dimension_index(name: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[name[-1]]


def _instruction_source_register_names(instruction: Instruction) -> set[str]:
    """Return normal register names read by ``instruction``.

    Special registers are scheduler-provided and therefore do not participate
    in the per-lane scoreboard.  Memory operands contribute their base register
    because the address must be ready before a load or store can issue.
    """
    sources: set[str] = set()

    def add_operand(operand: object) -> None:
        if isinstance(operand, Reg) and operand.name not in SPECIAL_REGISTER_NAMES:
            sources.add(operand.name)
        elif isinstance(operand, Mem) and operand.base not in SPECIAL_REGISTER_NAMES:
            sources.add(operand.base)

    if instruction.op in REGISTER_DEST_OPS or instruction.op in MEMORY_DEST_OPS:
        for operand in instruction.srcs:
            add_operand(operand)
        if instruction.op in MEMORY_DEST_OPS:
            add_operand(instruction.dst)
    elif instruction.op == "BRA":
        # A conditional BRA reads its condition register; labels are symbolic
        # and are resolved by the parser rather than the scoreboard.
        if len(instruction.operands) > 1:
            add_operand(instruction.operands[1])
    return sources


def _detect_shared_bank_conflicts(
    device: Device,
    sm: SM,
    warp: WarpState,
    instruction: Instruction,
) -> int:
    """Count the extra shared-memory transactions caused by bank conflicts.

    Shared memory is modeled as ``bank_count`` 4-byte banks.  Lanes that access
    the same bank in the same instruction produce one additional transaction
    for every lane after the first.  This is reported as a counter; the
    functional simulator still serializes the accesses correctly.
    """
    mem_operand = (
        instruction.dst if instruction.op in MEMORY_DEST_OPS else instruction.srcs[0]
    )
    if not isinstance(mem_operand, Mem):
        return 0

    bank_hits: dict[int, int] = {}
    for lane in warp.active_lanes:
        address = as_int(_read_operand(device, sm, warp, lane, mem_operand))
        bank = (address // 4) % device.timing_model.bank_count
        bank_hits[bank] = bank_hits.get(bank, 0) + 1

    return sum(max(count - 1, 0) for count in bank_hits.values())


def _read_operand(
    device: Device, sm: SM, warp: WarpState, lane: LaneState, operand: object
) -> int | float:
    """Resolve a parsed operand to a per-lane scalar value.

    Registers, special registers, immediates, and memory-address expressions
    all flow through this function.  Memory operands resolve to an absolute
    address for the caller to load/store.
    """
    if isinstance(operand, Reg):
        if operand.name in SPECIAL_REGISTER_NAMES:
            return _special_register_value(device, sm, warp, lane, operand.name)
        return lane.regs.read(operand.name)
    if isinstance(operand, Imm):
        return operand.value
    if isinstance(operand, Mem):
        # Memory operands are base-plus-byte-offset.  The base may be a normal
        # register or a scheduler-provided special register.
        base = (
            _special_register_value(device, sm, warp, lane, operand.base)
            if operand.base in SPECIAL_REGISTER_NAMES
            else lane.regs.read(operand.base)
        )
        return base + operand.offset
    raise ValueError(f"unsupported operand: {operand!r}")


def _write_register(lane: LaneState, dst: Reg, value: int | float) -> None:
    lane.regs.write(dst.name, value)


def _alu(
    op: str, a: int | float, b: int | float | None = None, c: int | float | None = None
) -> int | float:
    # This is the scalar ALU.  It is called once per active lane, which is the
    # simplest possible implementation of a vector/SIMT ALU.
    if op == "MOV":
        return a
    if op == "NEG":
        return -a
    if op == "ABS":
        return abs(a)
    if op == "RCP":
        return 1.0 / a
    if op == "RSQRT":
        return 1.0 / math.sqrt(a)
    if op == "EXP2":
        return math.exp2(a)
    if op == "LOG2":
        return math.log2(a)
    if op == "FMA":
        if b is None or c is None:
            raise ValueError("FMA requires two source operands")
        return a * b + c
    if b is None:
        raise ValueError(f"{op} requires a source operand")
    if op == "ADD":
        return a + b
    if op == "SUB":
        return a - b
    if op == "MUL":
        return a * b
    if op == "DIV":
        return a / b
    if op == "MAX":
        return max(a, b)
    if op == "MIN":
        return min(a, b)
    raise ValueError(f"unsupported ALU opcode: {op}")


def _apply_reconvergence(warp: WarpState) -> bool:
    """Handle arrival at a divergent-branch reconvergence point.

    Returns ``True`` when the warp PC or active mask changed and the caller
    should check again before fetching an instruction.
    """
    if not warp.reconvergence_stack:
        return False

    state = warp.reconvergence_stack[-1]
    if warp.pc != state.reconvergence_pc:
        return False

    if state.phase == "taken":
        # The originally taken path reached the join point.  Save its mask and
        # run the fallthrough side next.
        warp.active_mask = state.pending_mask
        warp.pc = state.pending_pc
        state.phase = "fallthrough"
        return True

    # The fallthrough side has also reached the join point.  Both sides are
    # complete, so restore the original mask and pop this branch state.
    warp.reconvergence_stack.pop()
    warp.active_mask = state.original_mask
    return True


def _execute_branch(
    device: Device,
    sm: SM,
    warp: WarpState,
    instruction: Instruction,
    program: Program,
) -> None:
    """Execute a BRA instruction.

    ``BRA target`` is unconditional.  ``BRA target, condition`` is a divergent
    branch.  ``BRA target, condition, reconvergence`` additionally supplies the
    explicit join label needed for if/else lowering.
    """
    operands = instruction.operands
    if not operands or not isinstance(operands[0], Label):
        raise ValueError("BRA requires a label target")

    target_pc = program.labels[operands[0].name]

    if len(operands) == 1:
        warp.pc = target_pc
        return

    condition_operand = operands[1]
    taken_mask = 0
    for lane in warp.active_lanes:
        if _read_operand(device, sm, warp, lane, condition_operand) != 0:
            taken_mask |= 1 << lane.lane_id

    fallthrough_mask = warp.active_mask & ~taken_mask

    if taken_mask == 0:
        # No lane takes the branch; the fallthrough mask already equals the
        # current active mask.  Advance PC directly because BRA does not fall
        # through the normal per-instruction PC increment in execute_warp.
        warp.pc += 1
        return

    if fallthrough_mask == 0:
        # All active lanes take the branch.  No divergence state is needed.
        warp.active_mask = taken_mask
        warp.pc = target_pc
        return

    # Both sides are active.  The compiler must provide an explicit join label.
    if len(operands) != 3 or not isinstance(operands[2], Label):
        raise ValueError("divergent BRA requires a reconvergence label")

    reconvergence_pc = program.labels[operands[2].name]
    warp.reconvergence_stack.append(
        ReconvergenceState(
            pending_pc=warp.pc + 1,
            pending_mask=fallthrough_mask,
            reconvergence_pc=reconvergence_pc,
            original_mask=warp.active_mask,
        )
    )
    warp.active_mask = taken_mask
    warp.pc = target_pc


def _execute_compare(
    device: Device,
    sm: SM,
    warp: WarpState,
    lane: LaneState,
    instruction: Instruction,
    op: str,
) -> None:
    """Write a 1.0/0.0 predicate register from a lane comparison."""
    if len(instruction.srcs) != 2:
        raise ValueError(f"{op} requires two source operands")
    a = _read_operand(device, sm, warp, lane, instruction.srcs[0])
    b = _read_operand(device, sm, warp, lane, instruction.srcs[1])

    condition = op[4:] if op.startswith("CMP.") else "EQ"
    if condition == "EQ":
        result = a == b
    elif condition == "NE":
        result = a != b
    elif condition == "LT":
        result = a < b
    elif condition == "LE":
        result = a <= b
    elif condition == "GT":
        result = a > b
    elif condition == "GE":
        result = a >= b
    else:
        raise ValueError(f"unsupported comparison: {op}")

    _write_register(lane, instruction.dst, 1.0 if result else 0.0)


def _execute_select(
    device: Device,
    sm: SM,
    warp: WarpState,
    lane: LaneState,
    instruction: Instruction,
) -> None:
    """Implement ``SEL Rd, Rcond, Rtrue, Rfalse``."""
    if len(instruction.srcs) != 3:
        raise ValueError("SEL requires condition, true value, and false value")
    condition = _read_operand(device, sm, warp, lane, instruction.srcs[0])
    true_value = _read_operand(device, sm, warp, lane, instruction.srcs[1])
    false_value = _read_operand(device, sm, warp, lane, instruction.srcs[2])
    _write_register(
        lane, instruction.dst, true_value if condition != 0 else false_value
    )


def _execute_barrier(sm: SM, warp: WarpState) -> None:
    """Implement a block-scoped ``BAR.SYNC``.

    A warp marks itself waiting and stalls.  When the last warp in the block
    arrives, all waiting warps are released and advance past the barrier.
    """
    warp.waiting_at_barrier = True

    if all(w.waiting_at_barrier for w in sm.warps):
        for waiting_warp in sm.warps:
            waiting_warp.waiting_at_barrier = False
            waiting_warp.pc += 1


def execute_warp(
    device: Device,
    sm: SM,
    warp: WarpState,
    instruction: Instruction,
    program: Program,
) -> None:
    """Execute one instruction for all active lanes in a warp.

    The function handles a few terminal/control opcodes directly, then falls
    through to per-lane memory or ALU behavior.  It increments the warp PC once
    at the end, mirroring lockstep SIMT execution.
    """
    op = instruction.op

    if op == "EXIT":
        warp.exited = True
        return
    if op == "RET":
        warp.exited = True
        return
    if op == "BAR.SYNC":
        _execute_barrier(sm, warp)
        return
    if op == "BRA":
        _execute_branch(device, sm, warp, instruction, program)
        return

    for lane in warp.active_lanes:
        if op in {"LDG", "LDS"}:
            # Loads read 4 bytes from either global or shared memory.  Each
            # lane computes its own address from its own register values.
            if not isinstance(instruction.dst, Reg):
                raise ValueError(f"{op} destination is not a register")
            mem_operand = instruction.srcs[0]
            if not isinstance(mem_operand, Mem):
                raise ValueError(f"{op} source is not a memory operand")
            addr = as_int(_read_operand(device, sm, warp, lane, mem_operand))
            memory = device.global_memory if op == "LDG" else sm.shared_memory
            value = memory.read_f32(addr)
            _write_register(lane, instruction.dst, value)
        elif op in {"STG", "STS"}:
            # Stores write the value from the source register to the resolved
            # per-lane address.  Shared-memory writes are made visible to all
            # warps in the block by the BAR.SYNC implementation above.
            mem_operand = instruction.dst
            if not isinstance(mem_operand, Mem):
                raise ValueError(f"{op} destination is not a memory operand")
            value = _read_operand(device, sm, warp, lane, instruction.srcs[0])
            addr = as_int(_read_operand(device, sm, warp, lane, mem_operand))
            memory = device.global_memory if op == "STG" else sm.shared_memory
            memory.write_f32(addr, float(value))
        elif op in {
            "ADD",
            "SUB",
            "MUL",
            "DIV",
            "MAX",
            "MIN",
            "FMA",
            "MOV",
            "NEG",
            "ABS",
            "RCP",
            "RSQRT",
            "EXP2",
            "LOG2",
        }:
            if op == "FMA":
                # FMA uses three source operands: a, b, and the accumulated
                # value c.  This maps directly to the future MMA/vector FMA
                # datapath.
                if len(instruction.srcs) != 3:
                    raise ValueError(
                        "FMA expects three source operands after destination"
                    )
                a = _read_operand(device, sm, warp, lane, instruction.srcs[0])
                b = _read_operand(device, sm, warp, lane, instruction.srcs[1])
                c = _read_operand(device, sm, warp, lane, instruction.srcs[2])
                value = _alu(op, a, b, c)
            else:
                a = _read_operand(device, sm, warp, lane, instruction.srcs[0])
                b = (
                    _read_operand(device, sm, warp, lane, instruction.srcs[1])
                    if len(instruction.srcs) > 1
                    else None
                )
                value = _alu(op, a, b)
            _write_register(lane, instruction.dst, value)
        elif op.startswith("CMP.") or op == "CMP":
            _execute_compare(device, sm, warp, lane, instruction, op)
        elif op == "SEL":
            _execute_select(device, sm, warp, lane, instruction)
        else:
            raise NotImplementedError(f"opcode not implemented yet: {op}")

    warp.pc += 1
