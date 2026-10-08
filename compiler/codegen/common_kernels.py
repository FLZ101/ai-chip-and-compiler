"""ISA code generation for the shared compute backbone.

Step 008 implements the first production MLIR/Model-IR to ISA lowerings:

* rank-1 elementwise add/sub/mul/div,
* rank-1 sum/mean/max reductions using shared-memory tree reduction,
* square tiled matrix multiplication using shared-memory tiles and ``MMA``.

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
    if (m, k, n) != (16, 16, 16):
        raise NotImplementedError(
            "fragment-aware matmul lowering currently supports one 16x16x16 tile"
        )

    return _emit_mma_matmul_isa(a, b, c)


def _emit_mma_matmul_isa(
    a: Any, b: Any, c: Any
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Emit a warp-wide 16x16x16 MMA matmul.

    The 32-lane warp loads A/B fragments into consecutive register blocks,
    executes one fragment-wide ``MMA`` instruction, and stores the C/D
    fragment back to global memory.

    Register allocation:

    * ``R0..R7``  -> A fragment
    * ``R8..R15`` -> B fragment
    * ``R16..R23`` -> C input and D output fragment
    * ``R24``     -> lane row
    * ``R25``     -> lane column
    * ``R26``     -> computed byte address
    * ``R27``     -> temporary index
    """
    lines: list[str] = [
        # lane = tid.x, row = lane // 4, col = lane % 4.
        "IDIV R24, tid.x, 4",
        "MUL R25, R24, 4",
        "SUB R25, tid.x, R25",
    ]

    # Zero the C/D fragment before accumulating the single K tile.
    for register in range(16, 24):
        lines.append(f"MOV R{register}, 0")

    # Load one A element and one B element into each fragment register slot.
    for slot in range(8):
        row_group = slot // 4
        col_group = slot % 4
        a_reg = slot
        b_reg = 8 + slot
        lines.extend(
            _emit_fragment_load(
                dest=a_reg,
                row_group=row_group,
                col_group=col_group,
                base=a.base_addr,
                stride=16,
            )
        )
        lines.extend(
            _emit_fragment_load(
                dest=b_reg,
                row_group=row_group,
                col_group=col_group,
                base=b.base_addr,
                stride=16,
            )
        )

    lines.append("MMA R16, R0, R8, R16")

    # Store the D fragment.  The C and D layouts are identical.
    for slot in range(8):
        row_group = slot // 4
        col_group = slot % 4
        lines.extend(
            _emit_fragment_store(
                source=16 + slot,
                row_group=row_group,
                col_group=col_group,
                base=c.base_addr,
                stride=16,
            )
        )

    lines.append("EXIT")
    return "\n".join(lines) + "\n", ((1,), (32,))


def _emit_fragment_load(
    *,
    dest: int,
    row_group: int,
    col_group: int,
    base: int,
    stride: int,
) -> list[str]:
    """Emit ``LDG Rd, [global_address(row, col)]`` for one fragment element."""
    return [
        f"MOV R26, {row_group * 8}",
        "ADD R26, R26, R24",
        f"MUL R26, R26, {stride}",
        f"MOV R27, {col_group * 4}",
        "ADD R27, R27, R25",
        "ADD R26, R26, R27",
        "MUL R26, R26, 4",
        f"ADD R26, R26, {base}",
        f"LDG R{dest}, [R26]",
    ]


def _emit_fragment_store(
    *,
    source: int,
    row_group: int,
    col_group: int,
    base: int,
    stride: int,
) -> list[str]:
    """Emit ``STG [global_address(row, col)], Rs`` for one fragment element."""
    return [
        f"MOV R26, {row_group * 8}",
        "ADD R26, R26, R24",
        f"MUL R26, R26, {stride}",
        f"MOV R27, {col_group * 4}",
        "ADD R27, R27, R25",
        "ADD R26, R26, R27",
        "MUL R26, R26, 4",
        f"ADD R26, R26, {base}",
        f"STG [R26], R{source}",
    ]


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()
