"""ISA lowering for Transformer normalization and softmax primitives."""

from __future__ import annotations

import math
from typing import Any

from ..model_ir import Graph, Op


def lower_transformer_kernel(
    graph: Graph,
    mlir_text: str,
    layout: dict[str, Any],
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Lower one Transformer norm/softmax node to simulator ISA."""
    del mlir_text

    if len(graph.nodes) != 1 or len(graph.outputs) != 1:
        raise NotImplementedError("Transformer lowering supports one output")

    node = graph.nodes[0]
    if node.op == Op.SOFTMAX:
        return _softmax_isa(node, layout)
    if node.op == Op.LAYER_NORM:
        return _layer_norm_isa(node, layout)
    if node.op == Op.RMS_NORM:
        return _rms_norm_isa(node, layout)
    raise NotImplementedError(f"no Transformer lowering for op {node.op.value!r}")


def _prologue(n: int) -> list[str]:
    return [
        "MOV R0, ctaid.x",
        "MUL R0, R0, ntid.x",
        "ADD R0, R0, tid.x",
        f"MOV R1, {n}",
        "CMP.LT R2, R0, R1",
        "BRA body, R2, end",
        "BRA end",
        "body:",
    ]


def _finish() -> list[str]:
    return ["BRA end", "end:", "EXIT"]


def _launch_1d(n: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    block_size = 32
    grid_x = math.ceil(n / block_size)
    return (grid_x,), (block_size,)


def _row_address(base: int, row_reg: str, d: int, dim: int) -> list[str]:
    return [
        f"MOV R2, {dim}",
        f"MUL R2, {row_reg}, R2",
        f"ADD R2, R2, {d}",
        "MUL R2, R2, 4",
        f"MOV R3, {base}",
        "ADD R2, R2, R3",
    ]


def _softmax_isa(
    node: Any, layout: dict[str, Any]
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    x = layout[node.inputs[0].name]
    output = layout[node.outputs[0].name]
    if len(x.shape) != 2 or x.shape != output.shape:
        raise NotImplementedError("softmax lowering supports rank-2 tensors")
    rows, dim = x.shape

    lines = _prologue(rows)
    lines.append("MOV R20, -1000000000.0")

    # Pass 1: max.
    for d in range(dim):
        lines.extend(_row_address(x.base_addr, "R0", d, dim))
        lines.extend(
            [
                "LDG R4, [R2]",
                "MAX R20, R20, R4",
            ]
        )

    # Pass 2: sum of exp(x - max).
    lines.append("MOV R21, 0")
    for d in range(dim):
        lines.extend(_row_address(x.base_addr, "R0", d, dim))
        lines.extend(
            [
                "LDG R4, [R2]",
                "SUB R4, R4, R20",
                "MUL R4, R4, 1.4426950408889634",
                "EXP2 R5, R4",
                "ADD R21, R21, R5",
            ]
        )

    # Pass 3: write exp / sum.
    for d in range(dim):
        lines.extend(_row_address(x.base_addr, "R0", d, dim))
        lines.extend(
            [
                "LDG R4, [R2]",
                "SUB R4, R4, R20",
                "MUL R4, R4, 1.4426950408889634",
                "EXP2 R5, R4",
                "RCP R6, R21",
                "MUL R5, R5, R6",
            ]
        )
        lines.extend(_row_address(output.base_addr, "R0", d, dim))
        lines.append("STG [R2], R5")

    lines.extend(_finish())
    return "\n".join(lines) + "\n", _launch_1d(rows)


def _layer_norm_isa(
    node: Any, layout: dict[str, Any]
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    x = layout[node.inputs[0].name]
    weight = layout[node.inputs[1].name]
    bias = layout[node.inputs[2].name]
    output = layout[node.outputs[0].name]
    if len(x.shape) != 2 or x.shape != output.shape:
        raise NotImplementedError("layer_norm lowering supports rank-2 tensors")
    rows, dim = x.shape
    eps = float(node.attrs.get("epsilon", 1e-5))

    lines = _prologue(rows)
    lines.extend(["MOV R20, 0", "MOV R21, 0"])

    # First pass: sum and sum of squares.
    for d in range(dim):
        lines.extend(_row_address(x.base_addr, "R0", d, dim))
        lines.extend(
            [
                "LDG R4, [R2]",
                "ADD R20, R20, R4",
                "MUL R5, R4, R4",
                "ADD R21, R21, R5",
            ]
        )

    lines.extend(
        [
            f"MOV R2, {dim}",
            "DIV R22, R20, R2",
            "DIV R23, R21, R2",
            "MUL R24, R22, R22",
            "SUB R23, R23, R24",
            f"ADD R23, R23, {eps}",
            "RSQRT R24, R23",
        ]
    )

    # Second pass: normalize and scale.
    for d in range(dim):
        lines.extend(_row_address(x.base_addr, "R0", d, dim))
        lines.extend(
            [
                "LDG R4, [R2]",
                "SUB R4, R4, R22",
                "MUL R4, R4, R24",
            ]
        )
        lines.extend(
            [
                f"MOV R5, {d}",
                "MUL R5, R5, 4",
                f"MOV R6, {weight.base_addr}",
                "ADD R5, R5, R6",
                "LDG R7, [R5]",
                "MUL R4, R4, R7",
                f"MOV R5, {d}",
                "MUL R5, R5, 4",
                f"MOV R6, {bias.base_addr}",
                "ADD R5, R5, R6",
                "LDG R7, [R5]",
                "ADD R4, R4, R7",
            ]
        )
        lines.extend(_row_address(output.base_addr, "R0", d, dim))
        lines.append("STG [R2], R4")

    lines.extend(_finish())
    return "\n".join(lines) + "\n", _launch_1d(rows)


def _rms_norm_isa(
    node: Any, layout: dict[str, Any]
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    x = layout[node.inputs[0].name]
    weight = layout[node.inputs[1].name]
    output = layout[node.outputs[0].name]
    if len(x.shape) != 2 or x.shape != output.shape:
        raise NotImplementedError("rms_norm lowering supports rank-2 tensors")
    rows, dim = x.shape
    eps = float(node.attrs.get("epsilon", 1e-5))

    lines = _prologue(rows)
    lines.append("MOV R20, 0")

    for d in range(dim):
        lines.extend(_row_address(x.base_addr, "R0", d, dim))
        lines.extend(
            [
                "LDG R4, [R2]",
                "MUL R4, R4, R4",
                "ADD R20, R20, R4",
            ]
        )

    lines.extend(
        [
            f"MOV R2, {dim}",
            "DIV R21, R20, R2",
            f"ADD R21, R21, {eps}",
            "RSQRT R22, R21",
        ]
    )

    for d in range(dim):
        lines.extend(_row_address(x.base_addr, "R0", d, dim))
        lines.extend(
            [
                "LDG R4, [R2]",
                "MUL R4, R4, R22",
            ]
        )
        lines.extend(
            [
                f"MOV R5, {d}",
                "MUL R5, R5, 4",
                f"MOV R6, {weight.base_addr}",
                "ADD R5, R5, R6",
                "LDG R7, [R5]",
                "MUL R4, R4, R7",
            ]
        )
        lines.extend(_row_address(output.base_addr, "R0", d, dim))
        lines.append("STG [R2], R4")

    lines.extend(_finish())
    return "\n".join(lines) + "\n", _launch_1d(rows)
