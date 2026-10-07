"""ISA lowering for elementwise activation kernels."""

from __future__ import annotations

import math
from typing import Any

from ..model_ir import Graph, Op


def lower_activation_kernel(
    graph: Graph,
    mlir_text: str,
    layout: dict[str, Any],
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Lower one unary activation node to simulator ISA."""
    del mlir_text

    if len(graph.nodes) != 1 or len(graph.outputs) != 1:
        raise NotImplementedError("activation lowering supports one output")

    node = graph.nodes[0]
    value = layout[node.inputs[0].name]
    output = layout[node.outputs[0].name]
    if len(value.shape) != 1 or value.shape != output.shape:
        raise NotImplementedError("activation lowering supports rank-1 tensors")

    n = value.shape[0]
    block_size = 32
    grid_x = math.ceil(n / block_size)
    output_delta = output.base_addr - value.base_addr

    lines = [
        "MOV R0, ctaid.x",
        "MUL R0, R0, ntid.x",
        "ADD R0, R0, tid.x",
        f"MOV R1, {n}",
        "CMP.LT R2, R0, R1",
        "BRA body, R2, end",
        "BRA end",
        "body:",
        "MUL R3, R0, 4",
        f"MOV R4, {value.base_addr}",
        "ADD R3, R3, R4",
        "LDG R5, [R3]",
    ]
    lines.extend(_activation_body(node))
    lines.extend(
        [
            f"STG [R3 + {output_delta}], R6",
            "BRA end",
            "end:",
            "EXIT",
        ]
    )

    return "\n".join(lines) + "\n", ((grid_x,), (block_size,))


def _activation_body(node: Any) -> list[str]:
    op = node.op
    if op == Op.RELU:
        return ["MAX R6, R5, 0"]

    if op == Op.LEAKY_RELU:
        alpha = float(node.attrs.get("alpha", 0.01))
        return [
            "CMP.LT R7, R5, 0",
            f"MUL R8, R5, {alpha}",
            "SEL R6, R7, R8, R5",
        ]

    if op == Op.SIGMOID:
        return _sigmoid_sequence("R5", "R6", start=7)

    if op == Op.SILU:
        lines = _sigmoid_sequence("R5", "R6", start=7)
        lines.append("MUL R6, R5, R6")
        return lines

    if op == Op.GELU:
        return _gelu_sequence()

    raise NotImplementedError(f"no activation lowering for op {op.value!r}")


def _sigmoid_sequence(src: str, dst: str, *, start: int) -> list[str]:
    # sigmoid(x) = 1 / (1 + 2^(-x / ln(2))).
    r0 = f"R{start}"
    r1 = f"R{start + 1}"
    r2 = f"R{start + 2}"
    r3 = f"R{start + 3}"
    return [
        f"MUL {r0}, {src}, -1.4426950408889634",
        f"EXP2 {r1}, {r0}",
        f"MOV {r2}, 1",
        f"ADD {r3}, {r1}, {r2}",
        f"RCP {dst}, {r3}",
    ]


def _gelu_sequence() -> list[str]:
    # GELU approximation used by the NumPy golden reference:
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    return [
        "MUL R7, R5, R5",
        "MUL R8, R7, R5",
        "MUL R8, R8, 0.044715",
        "ADD R8, R8, R5",
        "MUL R8, R8, 0.7978845608028654",
        "MUL R9, R8, 2",
        "MUL R9, R9, -1.4426950408889634",
        "EXP2 R10, R9",
        "MOV R11, 1",
        "ADD R12, R10, R11",
        "RCP R13, R12",
        "MUL R13, R13, 2",
        "MOV R14, 1",
        "SUB R13, R13, R14",
        "MOV R14, 1",
        "ADD R14, R14, R13",
        "MUL R6, R5, R14",
        "MUL R6, R6, 0.5",
    ]
