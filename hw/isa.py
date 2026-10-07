"""Minimal GPU instruction set architecture.

The MVP keeps the ISA deliberately small but includes the opcodes required by
both the CNN and Transformer/LLM paths.  The parser accepts a compact assembly
syntax that is intentionally close to what the MLIR backend will emit later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

SPECIAL_REGISTERS = {
    "tid",
    "ntid",
    "ctaid",
    "nctaid",
    "laneid",
    "warpid",
    "clock",
}

REGISTER_DEST_OPS = {
    "MOV",
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "FMA",
    "NEG",
    "ABS",
    "MAX",
    "MIN",
    "RCP",
    "RSQRT",
    "EXP2",
    "LOG2",
    "CMP",
    "SEL",
    "LDG",
    "LDS",
}

MEMORY_DEST_OPS = {"STG", "STS"}
CONTROL_OPS = {"BRA", "RET", "EXIT", "BAR.SYNC"}


@dataclass(frozen=True)
class Reg:
    name: str


@dataclass(frozen=True)
class Imm:
    value: int | float


@dataclass(frozen=True)
class Mem:
    base: str
    offset: int = 0


Operand = Reg | Imm | Mem


@dataclass(frozen=True)
class Instruction:
    op: str
    operands: tuple[Operand, ...] = ()

    @property
    def dst(self) -> Operand:
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
    return token in SPECIAL_REGISTERS or (token.startswith("R") and token[1:].isdigit())


def _parse_operand(token: str) -> Operand:
    token = token.strip()
    if not token:
        raise ValueError("empty operand")

    if token.startswith("[") and token.endswith("]"):
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
            raise ValueError(f"invalid operand: {token!r}") from exc


def _split_operands(text: str) -> list[str]:
    """Split top-level commas while preserving memory brackets."""
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
    instructions: list[Instruction] = []
    labels: dict[str, int] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";") or line.startswith("//"):
            continue

        # Strip trailing comments outside brackets.
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

    return Program(instructions, labels)


def _format_operand(operand: Operand) -> str:
    if isinstance(operand, Reg):
        return operand.name
    if isinstance(operand, Imm):
        if isinstance(operand.value, float):
            return repr(operand.value)
        return str(operand.value)
    if operand.offset:
        sign = "+" if operand.offset > 0 else "-"
        return f"[{operand.base} {sign} {abs(operand.offset)}]"
    return f"[{operand.base}]"


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
