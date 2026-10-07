"""ISA lowering for Transformer attention, RoPE, and KV-cache."""

from __future__ import annotations

import math
from typing import Any

from ..model_ir import Graph, Op


def lower_attention_kernel(
    graph: Graph,
    mlir_text: str,
    layout: dict[str, Any],
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Lower one attention/sequence node to simulator ISA."""
    del mlir_text

    if len(graph.nodes) != 1 or len(graph.outputs) != 1:
        raise NotImplementedError("attention lowering supports one output")

    node = graph.nodes[0]
    if node.op == Op.ROPE:
        return _rope_isa(node, layout)
    if node.op == Op.CAUSAL_MASK:
        return _causal_mask_isa(node, layout)
    if node.op == Op.SCALED_DOT_PRODUCT_ATTENTION:
        return _attention_isa(node, layout, masked=False)
    if node.op == Op.MASKED_ATTENTION:
        return _attention_isa(node, layout, masked=True)
    if node.op == Op.KV_CACHE_APPEND:
        return _kv_cache_append_isa(node, layout)
    if node.op == Op.KV_CACHE_GATHER:
        return _kv_cache_gather_isa(node, layout)
    raise NotImplementedError(f"no attention lowering for op {node.op.value!r}")


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


def _causal_mask_isa(
    node: Any, layout: dict[str, Any]
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    output = layout[node.outputs[0].name]
    if len(output.shape) != 4 or output.shape[0] != 1:
        raise NotImplementedError("causal_mask lowering supports shape (1,1,Q,K)")
    batch, heads, q_len, k_len = output.shape
    total = batch * heads * q_len * k_len

    lines = _prologue(total)
    lines.extend(
        [
            f"IDIV R1, R0, {k_len}",
            f"MUL R2, R1, {k_len}",
            "SUB R3, R0, R2",
            "CMP.GE R4, R1, R3",
            "MUL R5, R0, 4",
            f"MOV R6, {output.base_addr}",
            "ADD R5, R5, R6",
            "STG [R5], R4",
        ]
    )
    lines.extend(_finish())
    return "\n".join(lines) + "\n", _launch_1d(total)


def _rope_isa(
    node: Any, layout: dict[str, Any]
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    x = layout[node.inputs[0].name]
    cos = layout[node.inputs[1].name]
    sin = layout[node.inputs[2].name]
    output = layout[node.outputs[0].name]
    if len(x.shape) != 2 or x.shape != output.shape:
        raise NotImplementedError("rope lowering supports rank-2 tensors")
    rows, dim = x.shape
    total = rows * dim

    lines = _prologue(total)
    lines.extend(
        [
            f"IDIV R1, R0, {dim}",
            f"MUL R2, R1, {dim}",
            "SUB R3, R0, R2",
            "IDIV R4, R3, 2",
            "MUL R5, R4, 2",
            "SUB R6, R3, R5",
            f"MOV R7, {dim}",
            "MUL R7, R1, R7",
            "ADD R7, R7, R5",
            "MUL R7, R7, 4",
            f"MOV R8, {x.base_addr}",
            "ADD R7, R7, R8",
            "LDG R9, [R7]",
            "ADD R10, R7, 4",
            "LDG R11, [R10]",
            "SUB R12, R7, R8",
            f"MOV R13, {cos.base_addr}",
            "ADD R12, R12, R13",
            "LDG R14, [R12]",
            "SUB R12, R7, R8",
            f"MOV R13, {sin.base_addr}",
            "ADD R12, R12, R13",
            "LDG R15, [R12]",
            "MUL R16, R9, R14",
            "MUL R17, R11, R15",
            "SUB R18, R16, R17",
            "MUL R19, R9, R15",
            "MUL R20, R11, R14",
            "ADD R21, R19, R20",
            "CMP.EQ R22, R6, 0",
            "SEL R23, R22, R18, R21",
            "MUL R24, R0, 4",
            f"MOV R25, {output.base_addr}",
            "ADD R24, R24, R25",
            "STG [R24], R23",
        ]
    )
    lines.extend(_finish())
    return "\n".join(lines) + "\n", _launch_1d(total)


def _kv_cache_append_isa(
    node: Any, layout: dict[str, Any]
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    cache = layout[node.inputs[0].name]
    value = layout[node.inputs[1].name]
    output = layout[node.outputs[0].name]
    if cache.shape != value.shape or cache.shape != output.shape:
        raise NotImplementedError(
            "kv_cache_append lowering currently requires equal cache/value shapes"
        )
    n = output.numel
    block_size = 32
    grid_x = math.ceil(n / block_size)
    lines = _prologue(n)
    lines.extend(
        [
            "MUL R3, R0, 4",
            f"MOV R4, {value.base_addr}",
            "ADD R3, R3, R4",
            "LDG R5, [R3]",
            f"STG [R3 + {output.base_addr - value.base_addr}], R5",
        ]
    )
    lines.extend(_finish())
    return "\n".join(lines) + "\n", ((grid_x,), (block_size,))


def _kv_cache_gather_isa(
    node: Any, layout: dict[str, Any]
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    cache = layout[node.inputs[0].name]
    positions = layout[node.inputs[1].name]
    output = layout[node.outputs[0].name]
    if len(cache.shape) != 4 or len(output.shape) != 4:
        raise NotImplementedError("kv_cache_gather supports rank-4 cache/output")
    if cache.shape[0] != 1 or cache.shape[1] != 1:
        raise NotImplementedError("kv_cache_gather supports B=1,H=1")
    batch, heads, seq, dim = output.shape
    if batch != 1 or heads != 1:
        raise NotImplementedError("kv_cache_gather supports B=1,H=1")
    total = seq * dim
    lines = _prologue(total)
    # This first version supports one query position.
    lines.extend(
        [
            f"IDIV R1, R0, {dim}",
            f"MUL R2, R1, {dim}",
            "SUB R3, R0, R2",
            "MOV R4, 0",
            "MUL R4, R4, 4",
            f"MOV R5, {positions.base_addr}",
            "ADD R4, R4, R5",
            "LDG R6, [R4]",
            f"MOV R7, {dim}",
            "MUL R7, R6, R7",
            "ADD R7, R7, R3",
            "MUL R7, R7, 4",
            f"MOV R8, {cache.base_addr}",
            "ADD R7, R7, R8",
            "LDG R9, [R7]",
            "MUL R10, R0, 4",
            f"MOV R11, {output.base_addr}",
            "ADD R10, R10, R11",
            "STG [R10], R9",
        ]
    )
    lines.extend(_finish())
    return "\n".join(lines) + "\n", _launch_1d(total)


def _attention_isa(
    node: Any, layout: dict[str, Any], *, masked: bool
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    q = layout[node.inputs[0].name]
    k = layout[node.inputs[1].name]
    v = layout[node.inputs[2].name]
    mask = layout[node.inputs[3].name] if masked else None
    output = layout[node.outputs[0].name]
    if len(q.shape) != 4 or q.shape != k.shape or q.shape != v.shape:
        raise NotImplementedError("attention lowering supports rank-4 Q/K/V")
    if q.shape[0] != 1 or q.shape[1] != 1:
        raise NotImplementedError("attention lowering supports B=1,H=1")
    batch, heads, seq, dim = q.shape
    if seq != 2 or dim != 2:
        raise NotImplementedError("attention lowering currently supports S=2,D=2")
    scale = float(node.attrs.get("scale", 1.0 / math.sqrt(dim)))

    lines = _prologue(seq)
    lines.extend(["MOV R20, -1000000000.0", "MOV R21, 0"])

    # Pass 1: max score.
    for k_idx in range(seq):
        lines.extend(_attention_score(q, k, mask, "R0", k_idx, dim, seq, scale, masked))
        lines.append("MAX R20, R20, R2")

    # Pass 2: sum exp.
    for k_idx in range(seq):
        lines.extend(_attention_score(q, k, mask, "R0", k_idx, dim, seq, scale, masked))
        lines.extend(
            [
                "SUB R2, R2, R20",
                "MUL R2, R2, 1.4426950408889634",
                "EXP2 R3, R2",
                "ADD R21, R21, R3",
            ]
        )

    # Pass 3: output.
    lines.extend(["MOV R30, 0", "MOV R31, 0"])
    for k_idx in range(seq):
        lines.extend(_attention_score(q, k, mask, "R0", k_idx, dim, seq, scale, masked))
        lines.extend(
            [
                "SUB R2, R2, R20",
                "MUL R2, R2, 1.4426950408889634",
                "EXP2 R3, R2",
                "DIV R3, R3, R21",
            ]
        )
        for d in range(dim):
            lines.extend(
                [
                    f"MOV R4, {k_idx * dim + d}",
                    "MUL R4, R4, 4",
                    f"MOV R5, {v.base_addr}",
                    "ADD R4, R4, R5",
                    "LDG R6, [R4]",
                    "MUL R6, R3, R6",
                    f"ADD R{30 + d}, R{30 + d}, R6",
                ]
            )

    # Store output row.
    for d in range(dim):
        lines.extend(
            [
                f"MOV R4, {dim}",
                "MUL R4, R0, R4",
                f"ADD R4, R4, {d}",
                "MUL R4, R4, 4",
                f"MOV R5, {output.base_addr}",
                "ADD R4, R4, R5",
                f"STG [R4], R{30 + d}",
            ]
        )

    lines.extend(_finish())
    return "\n".join(lines) + "\n", _launch_1d(seq)


def _attention_score(
    q: Any,
    k: Any,
    mask: Any,
    q_reg: str,
    k_idx: int,
    dim: int,
    seq: int,
    scale: float,
    masked: bool,
) -> list[str]:
    del seq
    lines = ["MOV R2, 0"]
    for d in range(dim):
        lines.extend(
            [
                f"MOV R3, {d}",
                f"MOV R4, {dim}",
                f"MUL R4, {q_reg}, R4",
                "ADD R4, R4, R3",
                "MUL R4, R4, 4",
                f"MOV R5, {q.base_addr}",
                "ADD R4, R4, R5",
                "LDG R6, [R4]",
                f"MOV R7, {k_idx * dim + d}",
                "MUL R7, R7, 4",
                f"MOV R8, {k.base_addr}",
                "ADD R7, R7, R8",
                "LDG R9, [R7]",
                "FMA R2, R6, R9, R2",
            ]
        )
    lines.extend([f"MUL R2, R2, {scale}"])
    if masked:
        lines.extend(
            [
                f"MOV R3, {k_idx}",
                f"MOV R4, {dim}",
                f"MUL R4, {q_reg}, R4",
                "ADD R4, R4, R3",
                "MUL R4, R4, 4",
                f"MOV R5, {mask.base_addr}",
                "ADD R4, R4, R5",
                "LDG R6, [R4]",
                "CMP.EQ R7, R6, 0",
                "MOV R8, -1000000000.0",
                "SEL R2, R7, R8, R2",
            ]
        )
    return lines
