"""Minimal SIMT execution core."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .isa import Imm, Instruction, Mem, Program, Reg
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
    lane_id: int
    regs: RegisterFile = field(default_factory=RegisterFile)
    active: bool = True


@dataclass
class WarpState:
    warp_id: int
    lanes: list[LaneState]
    pc: int = 0
    exited: bool = False
    active_mask: int = 0xFFFFFFFF

    @property
    def active_lanes(self) -> list[LaneState]:
        return [
            lane
            for lane in self.lanes
            if lane.active and self.active_mask & (1 << lane.lane_id)
        ]


@dataclass
class SM:
    sm_id: int
    shared_memory: ByteMemory
    warps: list[WarpState] = field(default_factory=list)
    clock: int = 0


@dataclass
class Device:
    global_memory: ByteMemory
    sms: list[SM]
    grid_dim: tuple[int, ...] = (1,)
    block_dim: tuple[int, ...] = (1,)
    block_idx: tuple[int, ...] = (0,)
    clock: int = 0

    def run(self, program: Program) -> None:
        while True:
            self.clock += 1
            for sm in self.sms:
                sm.clock = self.clock
                for warp in sm.warps:
                    if warp.exited:
                        continue
                    if warp.pc >= len(program):
                        warp.exited = True
                        continue
                    instruction = program[warp.pc]
                    execute_warp(self, sm, warp, instruction)

            if all(warp.exited for sm in self.sms for warp in sm.warps):
                break


def _make_lanes(block_threads: int) -> list[LaneState]:
    return [LaneState(lane_id=i) for i in range(block_threads)]


def create_device(
    global_size: int = 1 << 20,
    shared_size: int = 1 << 16,
    sm_count: int = 4,
) -> Device:
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
    result: list[int] = []
    for dim in reversed(shape):
        result.append(index % dim)
        index //= dim
    return tuple(reversed(result))


def _special_register_value(
    device: Device, sm: SM, warp: WarpState, lane: LaneState, name: str
) -> int:
    if name == "laneid":
        return lane.lane_id
    if name == "warpid":
        return warp.warp_id
    if name == "tid":
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
    value = 0
    stride = 1
    for idx, dim in zip(reversed(index), reversed(shape)):
        value += idx * stride
        stride *= dim
    return value


def _read_operand(
    device: Device, sm: SM, warp: WarpState, lane: LaneState, operand: object
) -> int | float:
    if isinstance(operand, Reg):
        if operand.name in SPECIAL_REGISTER_NAMES:
            return _special_register_value(device, sm, warp, lane, operand.name)
        return lane.regs.read(operand.name)
    if isinstance(operand, Imm):
        return operand.value
    if isinstance(operand, Mem):
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


def execute_warp(
    device: Device, sm: SM, warp: WarpState, instruction: Instruction
) -> None:
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
        raise NotImplementedError("BRA is reserved for the control-flow milestone")

    for lane in warp.active_lanes:
        if op in {"LDG", "LDS"}:
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
        else:
            raise NotImplementedError(f"opcode not implemented yet: {op}")

    warp.pc += 1
