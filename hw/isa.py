"""Minimal GPU instruction set architecture.

The MVP keeps the ISA deliberately small but includes the opcodes required by
both the CNN and Transformer/LLM paths.  The parser accepts a compact assembly
syntax that is intentionally close to what the MLIR backend will emit later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# Registers whose value is provided by the GPU scheduler rather than by a
# kernel instruction.  They are read-only from the perspective of ISA code.
SPECIAL_REGISTERS = {
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

# Opcodes whose first operand is a register destination.
REGISTER_DEST_OPS = {
    # Data movement.
    "MOV",
    # Integer/FP arithmetic.
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "IDIV",
    # Fused multiply-add: dst = a * b + c.
    "FMA",
    # Tensor-core-style matrix multiply-accumulate.  The MVP models one
    # fragment element as dst = a * b + c.
    "MMA",
    # Unary ALU operations.
    "NEG",
    "ABS",
    # Min/max used by reductions and activation code.
    "MAX",
    "MIN",
    # Special functions used by softmax, LayerNorm, and RMSNorm.
    "RCP",
    "RSQRT",
    "EXP2",
    "LOG2",
    # Predicate/select operations reserved for control-flow lowering.
    "CMP",
    "CMP.EQ",
    "CMP.NE",
    "CMP.LT",
    "CMP.LE",
    "CMP.GT",
    "CMP.GE",
    "SEL",
    # Global and shared loads write a destination register.
    "LDG",
    "LDS",
    "LDG.F16",
    "LDS.F16",
    "LDG.BF16",
    "LDS.BF16",
    "LDG.I8",
    "LDS.I8",
    "LDG.U8",
    "LDS.U8",
    "LDG.I4",
    "LDS.I4",
    "LDG.U4",
    "LDS.U4",
    "LDG.I32",
    "LDS.I32",
}

# Opcodes whose first operand is a memory destination.  Memory stores use a
# different order from loads so the assembly remains close to typical SIMT ISAs:
#   STG [addr], value
MEMORY_DEST_OPS = {
    "STG",
    "STS",
    "STG.F16",
    "STS.F16",
    "STG.BF16",
    "STS.BF16",
    "STG.I8",
    "STS.I8",
    "STG.U8",
    "STS.U8",
    "STG.I4",
    "STS.I4",
    "STG.U4",
    "STS.U4",
    "STG.I32",
    "STS.I32",
}
# Opcodes that affect control flow.  Labels are resolved by the parser and
# used by the SIMT branch implementation in the execution core.
CONTROL_OPS = {"BRA", "RET", "EXIT", "BAR.SYNC"}


@dataclass(frozen=True)
class Reg:
    """A named register operand such as R0, tid.x, or laneid."""

    name: str


@dataclass(frozen=True)
class Imm:
    """An integer or floating-point immediate encoded directly in the instruction."""

    value: int | float


@dataclass(frozen=True)
class Mem:
    """A base-plus-offset memory operand.

    The parser accepts forms like ``[R0]``, ``[R0 + 16]``, and ``[R0 - 4]``.
    The simulator resolves the base register and adds the byte offset.
    """

    base: str
    offset: int = 0


@dataclass(frozen=True)
class Label:
    """A symbolic branch target.

    Labels are resolved to instruction indices by :func:`parse_asm`.
    """

    name: str


Operand = Reg | Imm | Mem | Label


@dataclass(frozen=True)
class Instruction:
    """One parsed ISA instruction.

    Operands are kept in their original assembly order.  ``dst`` and ``srcs``
    provide an opcode-aware view used by the execution core.
    """

    op: str
    operands: tuple[Operand, ...] = ()

    @property
    def dst(self) -> Operand:
        # Register-writing instructions put the destination first.  Loads such
        # as LDG and LDS also put the destination register first.
        if self.op in REGISTER_DEST_OPS:
            if not isinstance(self.operands[0], Reg):
                raise ValueError(f"{self.op} requires a register destination")
            return self.operands[0]
        if self.op in MEMORY_DEST_OPS:
            if not isinstance(self.operands[0], Mem):
                raise ValueError(f"{self.op} requires a memory destination")
            return self.operands[0]
        raise ValueError(f"{self.op} has no destination operand")

    @property
    def srcs(self) -> tuple[Operand, ...]:
        # Memory stores put the destination memory operand first, so all
        # remaining operands are sources.  Register/load destinations behave
        # the same way.
        if self.op in REGISTER_DEST_OPS or self.op in MEMORY_DEST_OPS:
            return self.operands[1:]
        return ()


@dataclass
class Program:
    instructions: list[Instruction]
    labels: dict[str, int]

    def __len__(self) -> int:
        return len(self.instructions)

    def __getitem__(self, index: int) -> Instruction:
        return self.instructions[index]


def _is_register_name(token: str) -> bool:
    # R0..RN are the normal per-thread registers.  Special registers describe
    # the current thread, block, grid, warp, lane, and clock.
    return token in SPECIAL_REGISTERS or (token.startswith("R") and token[1:].isdigit())


def _parse_operand(token: str) -> Operand:
    token = token.strip()
    if not token:
        raise ValueError("empty operand")

    if token.startswith("[") and token.endswith("]"):
        # Memory operands are byte-addressed in the ISA.  An empty bracket is a
        # convenience for address zero, although normal kernels should use an
        # explicit base register.
        inner = token[1:-1].strip()
        if not inner:
            return Mem("RZ", 0)
        if "+" in inner:
            base, offset = inner.split("+", 1)
            return Mem(base.strip(), int(offset.strip(), 0))
        if "-" in inner:
            base, offset = inner.split("-", 1)
            return Mem(base.strip(), -int(offset.strip(), 0))
        return Mem(inner, 0)

    if _is_register_name(token):
        return Reg(token)

    try:
        return Imm(int(token, 0))
    except ValueError:
        try:
            return Imm(float(token))
        except ValueError as exc:
            if token.replace("_", "").replace(".", "").isalnum() and token[0].isalpha():
                return Label(token)
            raise ValueError(f"invalid operand: {token!r}") from exc


def _split_operands(text: str) -> list[str]:
    """Split top-level commas while preserving memory brackets.

    A simple ``text.split(",")`` would break ``[R0 + 8], R1`` incorrectly if
    future syntax ever uses commas inside brackets.  Tracking bracket depth
    keeps memory operands intact.
    """
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def parse_asm(text: str) -> Program:
    """Parse assembly text into a :class:`Program`.

    The parser is deliberately permissive for valid instructions and strict for
    malformed destinations.  Unsupported opcodes are accepted syntactically but
    fail later in the execution core, so the ISA can grow without changing the
    parser.
    """
    instructions: list[Instruction] = []
    labels: dict[str, int] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";") or line.startswith("//"):
            continue

        # Strip trailing comments outside brackets.  Semicolons inside a memory
        # operand are not treated as comments.
        stripped: list[str] = []
        depth = 0
        for ch in line:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            if ch == ";" and depth == 0:
                break
            stripped.append(ch)
        line = "".join(stripped).strip()
        if not line:
            continue

        if line.endswith(":"):
            # Labels are recorded against the next instruction index and are
            # resolved by the execution core when a BRA target is reached.
            label = line[:-1].strip()
            if label in labels:
                raise ValueError(f"duplicate label: {label}")
            labels[label] = len(instructions)
            continue

        if " " not in line and "\t" not in line:
            op = line
            operand_text = ""
        else:
            op, operand_text = line.split(None, 1)

        op = op.upper()
        if op in CONTROL_OPS and operand_text:
            operands = tuple(
                _parse_operand(tok) for tok in _split_operands(operand_text)
            )
        elif op in CONTROL_OPS:
            operands = ()
        elif operand_text:
            operands = tuple(
                _parse_operand(tok) for tok in _split_operands(operand_text)
            )
        else:
            raise ValueError(f"missing operands for {op}")

        instr = Instruction(op, operands)

        # Early validation makes malformed ISA input fail at parse time.
        if op in REGISTER_DEST_OPS:
            if not operands or not isinstance(operands[0], Reg):
                raise ValueError(f"{op} requires a register destination")
        elif op in MEMORY_DEST_OPS:
            if not operands or not isinstance(operands[0], Mem):
                raise ValueError(f"{op} requires a memory destination")

        instructions.append(instr)

    # Branch targets and reconvergence labels must exist by the time parsing
    # completes.  This keeps branch-address bugs at parse time rather than
    # during simulator execution.
    for instruction in instructions:
        if instruction.op == "BRA":
            for operand in instruction.operands:
                if isinstance(operand, Label) and operand.name not in labels:
                    raise ValueError(f"undefined branch label: {operand.name}")

    return Program(instructions, labels)


def _format_operand(operand: Operand) -> str:
    if isinstance(operand, Reg):
        return operand.name
    if isinstance(operand, Imm):
        if isinstance(operand.value, float):
            return repr(operand.value)
        return str(operand.value)
    if isinstance(operand, Label):
        return operand.name
    if operand.offset:
        # Normalize the offset so round-tripping is stable regardless of whether
        # the user wrote +8, -4, or an unsigned immediate.
        sign = "+" if operand.offset > 0 else "-"
        return f"[{operand.base} {sign} {abs(operand.offset)}]"
    return f"[{operand.base}]"
    raise TypeError(f"unsupported operand for formatting: {operand!r}")


def format_instruction(instruction: Instruction) -> str:
    if not instruction.operands:
        return instruction.op
    return f"{instruction.op} " + ", ".join(
        _format_operand(operand) for operand in instruction.operands
    )


def format_program(program: Program) -> str:
    lines: list[str] = []
    label_to_pc = {pc: name for name, pc in program.labels.items()}
    for pc, instruction in enumerate(program.instructions):
        label = label_to_pc.get(pc)
        if label:
            lines.append(f"{label}:")
        lines.append(format_instruction(instruction))
    return "\n".join(lines)
