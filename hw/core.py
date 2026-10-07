"""Minimal SIMT execution core."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .isa import Imm, Instruction, Label, Mem, Program, Reg
from .memory import ByteMemory, RegisterFile, as_int

SPECIAL_REGISTER_NAMES = {
    "tid",
    "ntid",
    "ctaid",
    "nctaid",
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


@dataclass
class WarpState:
    """State for a 32-lane warp.

    ``active_mask`` is the SIMT execution mask.  The MVP does not yet implement
    branch reconvergence, but the mask is already the seam where that behavior
    will be added.
    """

    warp_id: int
    lanes: list[LaneState]
    pc: int = 0
    exited: bool = False
    active_mask: int = 0xFFFFFFFF
    reconvergence_stack: list["ReconvergenceState"] = field(default_factory=list)

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

    Each SM owns private shared memory and zero or more resident warps.  The
    first MVP only keeps one warp per SM to simplify scheduling.
    """

    sm_id: int
    shared_memory: ByteMemory
    warps: list[WarpState] = field(default_factory=list)
    clock: int = 0


@dataclass
class Device:
    """Top-level simulated GPU device."""

    global_memory: ByteMemory
    sms: list[SM]
    grid_dim: tuple[int, ...] = (1,)
    block_dim: tuple[int, ...] = (1,)
    block_idx: tuple[int, ...] = (0,)
    clock: int = 0

    def run(self, program: Program) -> None:
        # A single global cycle loop is the simplest deterministic scheduler.
        # It advances every non-exited warp once per cycle.  A future timing
        # model will replace this with issue, memory, and barrier stalls.
        while True:
            self.clock += 1
            for sm in self.sms:
                sm.clock = self.clock
                for warp in sm.warps:
                    if warp.exited:
                        continue
                    # A warp may arrive at a reconvergence label from either
                    # side of a divergent branch.  Resolve those stack
                    # transitions before fetching the next instruction.
                    while _apply_reconvergence(warp):
                        pass
                    if warp.pc >= len(program):
                        warp.exited = True
                        continue
                    instruction = program[warp.pc]
                    execute_warp(self, sm, warp, instruction, program)

            if all(warp.exited for sm in self.sms for warp in sm.warps):
                break


def _make_lanes(block_threads: int) -> list[LaneState]:
    return [LaneState(lane_id=i) for i in range(block_threads)]


def create_device(
    global_size: int = 1 << 20,
    shared_size: int = 1 << 16,
    sm_count: int = 4,
) -> Device:
    # The default sizes are intentionally small so tests run quickly.  The
    # driver/runtime will eventually allocate tensors against these regions.
    return Device(
        global_memory=ByteMemory(global_size, name="global"),
        sms=[
            SM(sm_id=i, shared_memory=ByteMemory(shared_size, name=f"shared{i}"))
            for i in range(sm_count)
        ],
    )


def launch(
    device: Device,
    program: Program,
    *,
    grid_dim: tuple[int, ...] = (1,),
    block_dim: tuple[int, ...] = (1,),
) -> None:
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
        lanes = _make_lanes(block_threads)
        warp = WarpState(warp_id=0, lanes=lanes)
        sm.warps.append(warp)
        # Record the block this SM is executing.  The MVP uses one warp and
        # one block per SM, so special register evaluation can stay simple.
        sm._block_idx = _unflatten_index(block_linear_id, grid_dim)  # type: ignore[attr-defined]
        sm._block_linear_id = block_linear_id  # type: ignore[attr-defined]

    device.run(program)


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
    if name == "tid":
        # tid is the global thread index for the launched grid.  For one warp
        # per block, this is the block's first thread plus the lane id.
        block_linear_id = getattr(sm, "_block_linear_id", 0)
        return block_linear_id * math.prod(device.block_dim) + lane.lane_id
    if name == "ntid":
        return math.prod(device.block_dim)
    if name == "ctaid":
        return _flatten_index(getattr(sm, "_block_idx", (0,)), device.grid_dim)
    if name == "nctaid":
        return math.prod(device.grid_dim)
    if name == "clock":
        return device.clock
    raise ValueError(f"unknown special register: {name}")


def _flatten_index(index: tuple[int, ...], shape: tuple[int, ...]) -> int:
    # Convert a multi-dimensional index to its linear form in row-major order.
    value = 0
    stride = 1
    for idx, dim in zip(reversed(index), reversed(shape)):
        value += idx * stride
        stride *= dim
    return value


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
        # All warps in this MVP block execute on the same SM and there is one
        # warp per block.  In the real implementation this becomes a
        # block-scoped barrier with per-warp arrival masks.
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
            # per-lane address.  Shared-memory visibility after barriers is a
            # later milestone; the MVP executes a single warp per SM.
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
