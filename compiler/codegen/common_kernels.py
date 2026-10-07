"""ISA code generation for the shared compute backbone.

Step 008 implements the first production MLIR/Model-IR to ISA lowerings:

* rank-1 elementwise add/sub/mul/div,
* rank-1 sum/mean/max reductions using shared-memory tree reduction,
* square tiled matrix multiplication using shared-memory tiles and ``FMA``.

The code generators are deliberately simple and deterministic.  They are the
reference lowerings consumed by the shared operator E2E harness.
"""

from __future__ import annotations

import math
from typing import Any

from ..model_ir import Graph, Op


def lower_common_kernel(
    graph: Graph,
    mlir_text: str,
    layout: dict[str, Any],
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Lower one common Model IR node to simulator ISA."""
    del mlir_text  # The textual MLIR is used by later generic passes.

    if len(graph.nodes) != 1 or len(graph.outputs) != 1:
        raise NotImplementedError(
            "lower_common_kernel supports exactly one single-output node"
        )

    node = graph.nodes[0]
    if node.op in {Op.ADD, Op.SUB, Op.MUL, Op.DIV}:
        return _lower_elementwise(graph, layout)
    if node.op in {Op.REDUCE_SUM, Op.REDUCE_MEAN, Op.REDUCE_MAX}:
        return _lower_reduction(graph, layout)
    if node.op == Op.MATMUL:
        return _lower_tiled_matmul(graph, layout)
    raise NotImplementedError(f"no common-kernel lowering for op {node.op.value!r}")


def _single_buffer(layout: dict[str, Any], value: Any) -> Any:
    return layout[value.name]


def _elementwise_isa(
    node: Any,
    left: Any,
    right: Any,
    output: Any,
    n: int,
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    block_size = 32
    grid_x = math.ceil(n / block_size)
    arith = {
        Op.ADD: "ADD",
        Op.SUB: "SUB",
        Op.MUL: "MUL",
        Op.DIV: "DIV",
    }[node.op]

    left_delta = right.base_addr - left.base_addr
    output_delta = output.base_addr - left.base_addr
    isa = f"""
MOV R0, ctaid.x
MUL R0, R0, ntid.x
ADD R0, R0, tid.x
MOV R1, {n}
CMP.LT R2, R0, R1
BRA body, R2, end
BRA end
body:
MUL R3, R0, 4
MOV R4, {left.base_addr}
ADD R3, R3, R4
LDG R5, [R3]
LDG R6, [R3 + {left_delta}]
{arith} R7, R5, R6
STG [R3 + {output_delta}], R7
BRA end
end:
EXIT
"""
    return isa, ((grid_x,), (block_size,))


def _lower_elementwise(
    graph: Graph,
    layout: dict[str, Any],
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    node = graph.nodes[0]
    left = _single_buffer(layout, node.inputs[0])
    right = _single_buffer(layout, node.inputs[1])
    output = _single_buffer(layout, node.outputs[0])

    if left.shape != right.shape or left.shape != output.shape:
        raise NotImplementedError("elementwise lowering requires equal shapes")
    if left.shape != right.shape or left.shape != output.shape:
        raise NotImplementedError("elementwise lowering requires equal shapes")

    return _elementwise_isa(node, left, right, output, left.numel)


def _reduction_identity(op: Op) -> float:
    if op == Op.REDUCE_MAX:
        return -1.0e30
    return 0.0


def _reduction_arith(op: Op) -> str:
    if op == Op.REDUCE_MAX:
        return "MAX"
    return "ADD"


def _lower_reduction(
    graph: Graph,
    layout: dict[str, Any],
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    node = graph.nodes[0]
    value = _single_buffer(layout, node.inputs[0])
    output = _single_buffer(layout, node.outputs[0])
    if len(value.shape) != 1:
        raise NotImplementedError("reduction lowering supports rank-1 tensors")

    n = value.shape[0]
    block_size = _next_power_of_two(max(32, n))
    identity = _reduction_identity(node.op)
    arith = _reduction_arith(node.op)

    lines: list[str] = [
        "MOV R0, tid.x",
        "MUL R3, R0, 4",
        f"MOV R1, {n}",
        "CMP.LT R2, R0, R1",
        "BRA load, R2, load_end",
        "BRA load_skip",
        "load:",
        f"MOV R4, {value.base_addr}",
        "ADD R7, R3, R4",
        "LDG R5, [R7]",
        "MOV R4, R5",
        "BRA load_end",
        "load_skip:",
        f"MOV R4, {identity}",
        "BRA load_end",
        "load_end:",
        "STS [R3], R4",
        "BAR.SYNC",
    ]

    stride = block_size // 2
    while stride > 0:
        body = f"tree_{stride}_body"
        skip = f"tree_{stride}_skip"
        end = f"tree_{stride}_end"
        lines.extend(
            [
                f"MOV R1, {stride}",
                "CMP.LT R2, R0, R1",
                f"BRA {body}, R2, {end}",
                f"BRA {end}",
                f"{body}:",
                "MUL R5, R0, 4",
                "MOV R6, R0",
                "ADD R6, R6, R1",
                "MUL R6, R6, 4",
                "LDS R7, [R5]",
                "LDS R8, [R6]",
                f"{arith} R9, R7, R8",
                "STS [R5], R9",
                f"BRA {end}",
                f"{skip}:",
                f"BRA {end}",
                f"{end}:",
                "BAR.SYNC",
            ]
        )
        stride //= 2

    lines.extend(
        [
            "CMP.EQ R2, R0, 0",
            "BRA store, R2, store_end",
            "BRA store_end",
            "store:",
            "MUL R3, R0, 4",
            "LDS R4, [R3]",
        ]
    )
    if node.op == Op.REDUCE_MEAN:
        lines.append(f"DIV R4, R4, {n}")
    lines.extend(
        [
            f"MOV R5, {output.base_addr}",
            "STG [R5], R4",
            "BRA store_end",
            "store_end:",
            "EXIT",
        ]
    )

    return "\n".join(lines) + "\n", ((1,), (block_size,))


def _lower_tiled_matmul(
    graph: Graph,
    layout: dict[str, Any],
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    node = graph.nodes[0]
    a = _single_buffer(layout, node.inputs[0])
    b = _single_buffer(layout, node.inputs[1])
    c = _single_buffer(layout, node.outputs[0])

    if len(a.shape) != 2 or len(b.shape) != 2 or len(c.shape) != 2:
        raise NotImplementedError("matmul lowering supports rank-2 matrices")
    m, k = a.shape
    k_b, n = b.shape
    if k != k_b:
        raise NotImplementedError("matmul inner dimensions do not match")
    if m != n:
        raise NotImplementedError("matmul lowering currently requires square matrices")
    if k % n != 0:
        raise NotImplementedError("matmul K must be divisible by the square tile size")

    tile = n
    block_threads = m * n
    smem_b_offset = m * tile * 4

    lines: list[str] = [
        "MOV R0, tid.x",
        "MOV R1, tid.y",
        f"MOV R20, 0",
    ]

    for k_start in range(0, k, tile):
        k_base = k_start
        # Cooperative copy of one A tile into shared memory.
        lines.extend(
            [
                f"MUL R2, R1, {k}",
                f"MOV R3, {k_base}",
                "ADD R3, R3, R0",
                "ADD R2, R2, R3",
                "MUL R2, R2, 4",
                f"MOV R4, {a.base_addr}",
                "ADD R2, R2, R4",
                "LDG R5, [R2]",
                f"MUL R6, R1, {tile}",
                "ADD R6, R6, R0",
                "MUL R6, R6, 4",
                "STS [R6], R5",
            ]
        )

        # Cooperative copy of one B tile into shared memory.
        lines.extend(
            [
                f"MUL R2, R1, {n}",
                "ADD R2, R2, R0",
                "MUL R2, R2, 4",
                f"MOV R3, {smem_b_offset}",
                "ADD R2, R2, R3",
                f"MOV R4, {k_base}",
                "ADD R4, R4, R1",
                f"MUL R4, R4, {n}",
                "ADD R4, R4, R0",
                "MUL R4, R4, 4",
                f"MOV R5, {b.base_addr}",
                "ADD R4, R4, R5",
                "LDG R6, [R4]",
                "STS [R2], R6",
                "BAR.SYNC",
            ]
        )

        # Each thread computes one C element with FMA.
        lines.append(f"MOV R7, {smem_b_offset}")
        for inner in range(tile):
            lines.extend(
                [
                    f"MUL R2, R1, {tile}",
                    f"ADD R2, R2, {inner}",
                    "MUL R2, R2, 4",
                    "LDS R3, [R2]",
                    f"MOV R4, {inner * n}",
                    "ADD R4, R4, R0",
                    "MUL R4, R4, 4",
                    "ADD R4, R4, R7",
                    "LDS R5, [R4]",
                    "FMA R20, R3, R5, R20",
                ]
            )
        lines.append("BAR.SYNC")

    lines.extend(
        [
            f"MUL R2, R1, {n}",
            "ADD R2, R2, R0",
            "MUL R2, R2, 4",
            f"MOV R3, {c.base_addr}",
            "ADD R2, R2, R3",
            "STG [R2], R20",
            "EXIT",
        ]
    )

    return "\n".join(lines) + "\n", ((1,), (n, m))


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()
