"""ISA lowering for the initial shape and layout kernels."""

from __future__ import annotations

import math
from typing import Any

from ..model_ir import Graph, Op


def lower_shape_kernel(
    graph: Graph,
    mlir_text: str,
    layout: dict[str, Any],
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Lower one shape/layout Model IR node to simulator ISA."""
    del mlir_text

    if len(graph.nodes) != 1 or len(graph.outputs) != 1:
        raise NotImplementedError("shape lowering supports one output")

    node = graph.nodes[0]
    if node.op == Op.RESHAPE:
        return _reshape_isa(node, layout)
    if node.op == Op.TRANSPOSE:
        return _transpose_isa(node, layout)
    if node.op == Op.CONCAT:
        return _concat_isa(node, layout)
    if node.op == Op.SLICE:
        return _slice_isa(node, layout)
    if node.op == Op.PAD:
        return _pad_isa(node, layout)
    if node.op == Op.RESIZE:
        return _resize_isa(node, layout)
    raise NotImplementedError(f"no shape lowering for op {node.op.value!r}")


def _reshape_isa(
    node: Any, layout: dict[str, Any]
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    value = layout[node.inputs[0].name]
    output = layout[node.outputs[0].name]
    if value.numel != output.numel:
        raise NotImplementedError("reshape lowering requires equal element counts")
    return _linear_copy_isa(value, output)


def _transpose_isa(
    node: Any, layout: dict[str, Any]
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    value = layout[node.inputs[0].name]
    output = layout[node.outputs[0].name]
    if len(value.shape) != 2 or len(output.shape) != 2:
        raise NotImplementedError("transpose lowering supports rank-2 tensors")
    rows, cols = value.shape
    if output.shape != (cols, rows):
        raise NotImplementedError("transpose output shape does not match input")

    isa = f"""
MOV R0, tid.x
MOV R1, tid.y
MUL R2, R0, {cols}
ADD R2, R2, R1
MUL R2, R2, 4
MOV R3, {value.base_addr}
ADD R2, R2, R3
LDG R4, [R2]
MUL R5, R1, {rows}
ADD R5, R5, R0
MUL R5, R5, 4
MOV R6, {output.base_addr}
ADD R5, R5, R6
STG [R5], R4
EXIT
"""
    return isa, ((1,), (rows, cols))


def _concat_isa(
    node: Any, layout: dict[str, Any]
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    if len(node.inputs) != 2:
        raise NotImplementedError("concat lowering currently supports two inputs")
    first = layout[node.inputs[0].name]
    second = layout[node.inputs[1].name]
    output = layout[node.outputs[0].name]
    if len(first.shape) == 4 and len(second.shape) == 4 and len(output.shape) == 4:
        return _concat4d_isa(node, layout)
    if len(first.shape) != 1 or len(second.shape) != 1 or len(output.shape) != 1:
        raise NotImplementedError("concat lowering supports rank-1 tensors")

    first_len = first.shape[0]
    n = output.shape[0]
    block_size = 32
    grid_x = math.ceil(n / block_size)
    isa = f"""
MOV R0, ctaid.x
MUL R0, R0, ntid.x
ADD R0, R0, tid.x
MOV R1, {n}
CMP.LT R2, R0, R1
BRA body, R2, end
BRA end
body:
MOV R2, {first_len}
CMP.LT R3, R0, R2
BRA first, R3, end
BRA second
first:
MUL R4, R0, 4
MOV R5, {first.base_addr}
ADD R4, R4, R5
LDG R6, [R4]
BRA store
second:
MOV R4, R0
MOV R5, {first_len}
SUB R4, R4, R5
MUL R4, R4, 4
MOV R5, {second.base_addr}
ADD R4, R4, R5
LDG R6, [R4]
BRA store
store:
MUL R7, R0, 4
MOV R8, {output.base_addr}
ADD R7, R7, R8
STG [R7], R6
BRA end
end:
EXIT
"""
    return isa, ((grid_x,), (block_size,))


def _slice_isa(
    node: Any, layout: dict[str, Any]
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    value = layout[node.inputs[0].name]
    output = layout[node.outputs[0].name]
    if len(value.shape) != 1 or len(output.shape) != 1:
        raise NotImplementedError("slice lowering supports rank-1 tensors")

    starts = tuple(node.attrs.get("starts", (0,)))
    steps = tuple(node.attrs.get("steps", (1,)))
    if len(starts) != 1 or len(steps) != 1:
        raise NotImplementedError("slice lowering supports one axis")
    start = int(starts[0])
    step = int(steps[0])
    n = output.shape[0]
    block_size = 32
    grid_x = math.ceil(n / block_size)
    isa = f"""
MOV R0, ctaid.x
MUL R0, R0, ntid.x
ADD R0, R0, tid.x
MOV R1, {n}
CMP.LT R2, R0, R1
BRA body, R2, end
BRA end
body:
MOV R2, R0
MUL R2, R2, {step}
ADD R2, R2, {start}
MUL R2, R2, 4
MOV R3, {value.base_addr}
ADD R2, R2, R3
LDG R4, [R2]
MUL R5, R0, 4
MOV R6, {output.base_addr}
ADD R5, R5, R6
STG [R5], R4
BRA end
end:
EXIT
"""
    return isa, ((grid_x,), (block_size,))


def _pad_isa(
    node: Any, layout: dict[str, Any]
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    value = layout[node.inputs[0].name]
    output = layout[node.outputs[0].name]
    if len(value.shape) != 1 or len(output.shape) != 1:
        raise NotImplementedError("pad lowering supports rank-1 tensors")

    pads = tuple(node.attrs.get("pads", ((0, 0),)))
    if len(pads) != 1 or len(pads[0]) != 2:
        raise NotImplementedError("pad lowering supports one axis")
    before, after = int(pads[0][0]), int(pads[0][1])
    pad_value = float(node.attrs.get("value", 0.0))
    input_len = value.shape[0]
    n = output.shape[0]
    block_size = 32
    grid_x = math.ceil(n / block_size)
    isa = f"""
MOV R0, ctaid.x
MUL R0, R0, ntid.x
ADD R0, R0, tid.x
MOV R1, {n}
CMP.LT R2, R0, R1
BRA body, R2, end
BRA end
body:
MOV R3, {before}
SUB R4, R0, R3
CMP.GE R5, R4, 0
MOV R6, {input_len}
CMP.LT R7, R4, R6
MUL R8, R5, R7
MOV R9, 0
MAX R10, R4, R9
MOV R9, {input_len - 1}
MIN R10, R10, R9
MUL R10, R10, 4
MOV R11, {value.base_addr}
ADD R10, R10, R11
LDG R12, [R10]
MOV R13, {pad_value}
SEL R14, R8, R12, R13
MUL R10, R0, 4
MOV R11, {output.base_addr}
ADD R10, R10, R11
STG [R10], R14
BRA end
end:
EXIT
"""
    return isa, ((grid_x,), (block_size,))


def _resize_isa(
    node: Any, layout: dict[str, Any]
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    value = layout[node.inputs[0].name]
    output = layout[node.outputs[0].name]
    if len(value.shape) == 4 and len(output.shape) == 4:
        return _resize4d_isa(node, layout)
    if len(value.shape) != 1 or len(output.shape) != 1:
        raise NotImplementedError("resize lowering supports rank-1 tensors")

    input_len = value.shape[0]
    output_len = output.shape[0]
    n = output_len
    block_size = 32
    grid_x = math.ceil(n / block_size)
    isa = f"""
MOV R0, ctaid.x
MUL R0, R0, ntid.x
ADD R0, R0, tid.x
MOV R1, {n}
CMP.LT R2, R0, R1
BRA body, R2, end
BRA end
body:
MOV R2, R0
MUL R2, R2, {input_len}
MOV R3, {output_len}
IDIV R4, R2, R3
MUL R4, R4, 4
MOV R5, {value.base_addr}
ADD R4, R4, R5
LDG R6, [R4]
MUL R7, R0, 4
MOV R8, {output.base_addr}
ADD R7, R7, R8
STG [R7], R6
BRA end
end:
EXIT
"""
    return isa, ((grid_x,), (block_size,))


def _resize4d_isa(
    node: Any, layout: dict[str, Any]
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    value = layout[node.inputs[0].name]
    output = layout[node.outputs[0].name]
    if value.shape[0] != 1 or output.shape[0] != 1:
        raise NotImplementedError("resize4d lowering supports N=1")
    n, c, h, w = value.shape
    _, c_out, oh, ow = output.shape
    if c != c_out:
        raise NotImplementedError("resize4d channel counts must match")
    total = c * oh * ow
    block_size = 32
    grid_x = math.ceil(total / block_size)
    isa = f"""
MOV R0, ctaid.x
MUL R0, R0, ntid.x
ADD R0, R0, tid.x
MOV R1, {total}
CMP.LT R2, R0, R1
BRA body, R2, end
BRA end
body:
IDIV R1, R0, {oh * ow}
MUL R2, R1, {oh * ow}
SUB R3, R0, R2
IDIV R2, R3, {ow}
MUL R4, R2, {ow}
SUB R5, R3, R4
MUL R6, R2, {h}
IDIV R6, R6, {oh}
MUL R7, R5, {w}
IDIV R7, R7, {ow}
MOV R8, R1
MUL R8, R8, {h}
ADD R8, R8, R6
MUL R8, R8, {w}
ADD R8, R8, R7
MUL R8, R8, 4
MOV R9, {value.base_addr}
ADD R8, R8, R9
LDG R10, [R8]
MUL R11, R0, 4
MOV R12, {output.base_addr}
ADD R11, R11, R12
STG [R11], R10
BRA end
end:
EXIT
"""
    return isa, ((grid_x,), (block_size,))


def _concat4d_isa(
    node: Any, layout: dict[str, Any]
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    first = layout[node.inputs[0].name]
    second = layout[node.inputs[1].name]
    output = layout[node.outputs[0].name]
    axis = int(node.attrs.get("axis", 1))
    if axis != 1:
        raise NotImplementedError("concat4d lowering supports channel axis only")
    if first.shape[0] != 1 or second.shape[0] != 1:
        raise NotImplementedError("concat4d lowering supports N=1")
    c1 = first.shape[1]
    c2 = second.shape[1]
    h, w = first.shape[2], first.shape[3]
    if (h, w) != (second.shape[2], second.shape[3]):
        raise NotImplementedError("concat4d spatial shapes must match")
    total = (c1 + c2) * h * w
    hw = h * w
    block_size = 32
    grid_x = math.ceil(total / block_size)
    isa = f"""
MOV R0, ctaid.x
MUL R0, R0, ntid.x
ADD R0, R0, tid.x
MOV R1, {total}
CMP.LT R2, R0, R1
BRA body, R2, end
BRA end
body:
IDIV R1, R0, {hw}
MUL R2, R1, {hw}
SUB R3, R0, R2
MOV R2, {c1}
CMP.LT R4, R1, R2
BRA first, R4, end
BRA second
first:
MUL R5, R1, {hw}
ADD R5, R5, R3
MUL R5, R5, 4
MOV R6, {first.base_addr}
ADD R5, R5, R6
LDG R7, [R5]
BRA store
second:
MOV R5, R1
MOV R6, {c1}
SUB R5, R5, R6
MUL R5, R5, {hw}
ADD R5, R5, R3
MUL R5, R5, 4
MOV R6, {second.base_addr}
ADD R5, R5, R6
LDG R7, [R5]
BRA store
store:
MUL R8, R0, 4
MOV R9, {output.base_addr}
ADD R8, R8, R9
STG [R8], R7
BRA end
end:
EXIT
"""
    return isa, ((grid_x,), (block_size,))


def _linear_copy_isa(
    value: Any, output: Any
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    n = output.numel
    block_size = 32
    grid_x = math.ceil(n / block_size)
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
MOV R4, {value.base_addr}
ADD R3, R3, R4
LDG R5, [R3]
STG [R3 + {output.base_addr - value.base_addr}], R5
BRA end
end:
EXIT
"""
    return isa, ((grid_x,), (block_size,))
