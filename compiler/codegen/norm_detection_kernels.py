"""ISA lowering for CNN normalization and detection helpers."""

from __future__ import annotations

import math
from typing import Any

from ..model_ir import Graph, Op


def lower_norm_detection_kernel(
    graph: Graph,
    mlir_text: str,
    layout: dict[str, Any],
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Lower one normalization/detection node to simulator ISA."""
    del mlir_text

    if len(graph.nodes) != 1 or len(graph.outputs) != 1:
        raise NotImplementedError("norm/detection lowering supports one output")

    node = graph.nodes[0]
    if node.op == Op.BATCH_NORM:
        return _batch_norm_isa(node, layout)
    if node.op == Op.GROUP_NORM:
        return _group_norm_isa(node, layout)
    if node.op == Op.BOX_DECODE:
        return _box_decode_isa(node, layout)
    if node.op == Op.NMS:
        return _nms_isa(node, layout)
    raise NotImplementedError(f"no norm/detection lowering for op {node.op.value!r}")


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


def _batch_norm_isa(
    node: Any, layout: dict[str, Any]
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    x = layout[node.inputs[0].name]
    scale = layout[node.inputs[1].name]
    bias = layout[node.inputs[2].name]
    mean = layout[node.inputs[3].name]
    var = layout[node.inputs[4].name]
    output = layout[node.outputs[0].name]

    if x.shape[0] != 1 or len(x.shape) != 4:
        raise NotImplementedError("batch_norm lowering supports NCHW with N=1")
    n, c, h, w = x.shape
    eps = float(node.attrs.get("epsilon", 1e-5))
    hw = h * w
    total = n * c * hw

    lines = _prologue(total)
    lines.extend(
        [
            f"IDIV R1, R0, {hw}",
            f"MUL R3, R1, {hw}",
            "SUB R2, R0, R3",
            "MUL R3, R1, 4",
            f"MOV R4, {mean.base_addr}",
            "ADD R3, R3, R4",
            "LDG R5, [R3]",
            "MUL R6, R1, 4",
            f"MOV R7, {var.base_addr}",
            "ADD R6, R6, R7",
            "LDG R8, [R6]",
            f"ADD R8, R8, {eps}",
            "RSQRT R8, R8",
            "MUL R6, R0, 4",
            f"MOV R7, {x.base_addr}",
            "ADD R6, R6, R7",
            "LDG R9, [R6]",
            "SUB R9, R9, R5",
            "MUL R9, R9, R8",
            "MUL R10, R1, 4",
            f"MOV R11, {scale.base_addr}",
            "ADD R10, R10, R11",
            "LDG R12, [R10]",
            "MUL R9, R9, R12",
            "MUL R10, R1, 4",
            f"MOV R11, {bias.base_addr}",
            "ADD R10, R10, R11",
            "LDG R12, [R10]",
            "ADD R9, R9, R12",
            "MUL R10, R0, 4",
            f"MOV R11, {output.base_addr}",
            "ADD R10, R10, R11",
            "STG [R10], R9",
        ]
    )
    lines.extend(_finish())
    return "\n".join(lines) + "\n", _launch_1d(total)


def _group_norm_isa(
    node: Any, layout: dict[str, Any]
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    x = layout[node.inputs[0].name]
    weight = layout[node.inputs[1].name]
    bias = layout[node.inputs[2].name]
    output = layout[node.outputs[0].name]

    if x.shape[0] != 1 or len(x.shape) != 4:
        raise NotImplementedError("group_norm lowering supports NCHW with N=1")
    n, c, h, w = x.shape
    groups = int(node.attrs.get("num_groups", 1))
    if c % groups != 0:
        raise NotImplementedError("group_norm channels must divide into groups")
    channels_per_group = c // groups
    eps = float(node.attrs.get("epsilon", 1e-5))
    hw = h * w
    count = channels_per_group * hw
    total = c * hw

    lines = _prologue(total)
    lines.extend(
        [
            f"IDIV R1, R0, {hw}",
            f"MUL R3, R1, {hw}",
            "SUB R2, R0, R3",
            f"IDIV R26, R1, {channels_per_group}",
            "MOV R21, 0",
            "MOV R22, 0",
        ]
    )

    # Group statistics: unrolled over all channels/spatial positions and
    # branchlessly add only elements in this thread's group.
    for channel in range(c):
        group = channel // channels_per_group
        for ih in range(h):
            for iw in range(w):
                idx = channel * hw + ih * w + iw
                lines.extend(
                    [
                        f"MOV R3, {group}",
                        "CMP.EQ R4, R26, R3",
                        f"MOV R3, {idx}",
                        "MUL R3, R3, 4",
                        f"MOV R5, {x.base_addr}",
                        "ADD R3, R3, R5",
                        "LDG R6, [R3]",
                        "MUL R7, R6, R4",
                        "ADD R21, R21, R7",
                        "MUL R6, R6, R6",
                        "MUL R6, R6, R4",
                        "ADD R22, R22, R6",
                    ]
                )

    # mean and variance.
    lines.extend(
        [
            f"MOV R3, {count}",
            "DIV R23, R21, R3",
            "DIV R24, R22, R3",
            "MUL R25, R23, R23",
            "SUB R24, R24, R25",
            f"ADD R24, R24, {eps}",
            "RSQRT R24, R24",
        ]
    )

    # Output element at channel R1 and spatial offset R2.
    lines.extend(
        [
            f"IDIV R3, R2, {w}",
            f"MUL R4, R3, {w}",
            "SUB R5, R2, R4",
            f"MOV R6, {h}",
            "MUL R6, R1, R6",
            "ADD R6, R6, R3",
            f"MUL R6, R6, {w}",
            "ADD R6, R6, R5",
            "MUL R6, R6, 4",
            f"MOV R7, {x.base_addr}",
            "ADD R6, R6, R7",
            "LDG R8, [R6]",
            "SUB R8, R8, R23",
            "MUL R8, R8, R24",
            "MUL R9, R1, 4",
            f"MOV R10, {weight.base_addr}",
            "ADD R9, R9, R10",
            "LDG R11, [R9]",
            "MUL R8, R8, R11",
            "MUL R9, R1, 4",
            f"MOV R10, {bias.base_addr}",
            "ADD R9, R9, R10",
            "LDG R11, [R9]",
            "ADD R8, R8, R11",
            "MUL R9, R0, 4",
            f"MOV R10, {output.base_addr}",
            "ADD R9, R9, R10",
            "STG [R9], R8",
        ]
    )
    lines.extend(_finish())
    return "\n".join(lines) + "\n", _launch_1d(total)


def _box_decode_isa(
    node: Any, layout: dict[str, Any]
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    boxes = layout[node.inputs[0].name]
    deltas = layout[node.inputs[1].name]
    output = layout[node.outputs[0].name]
    if len(boxes.shape) != 2 or boxes.shape[1] != 4:
        raise NotImplementedError("box_decode lowering expects (N,4) boxes")
    if boxes.shape != deltas.shape:
        raise NotImplementedError("box_decode boxes and deltas must have same shape")

    total = boxes.shape[0] * 4
    lines = _prologue(total)
    lines.extend(
        [
            "IDIV R1, R0, 4",
            "MUL R2, R1, 4",
            "SUB R3, R0, R2",
        ]
    )

    # Load all four box components and deltas.
    for component in range(4):
        lines.extend(
            [
                f"MUL R4, R1, 4",
                f"ADD R4, R4, {component}",
                "MUL R4, R4, 4",
                f"MOV R5, {boxes.base_addr}",
                "ADD R4, R4, R5",
                "LDG R6, [R4]",
                f"MOV R20, R6",
                f"MUL R4, R1, 4",
                f"ADD R4, R4, {component}",
                "MUL R4, R4, 4",
                f"MOV R5, {deltas.base_addr}",
                "ADD R4, R4, R5",
                "LDG R7, [R4]",
                f"MOV R21, R7",
            ]
        )
        # The code above only retains component 3; replace with a simpler
        # per-component loading scheme below.
        break

    # Load components explicitly into R20..R27.
    lines.extend(
        [
            "MUL R4, R1, 4",
            "MOV R5, 0",
            "ADD R4, R4, R5",
            "MUL R4, R4, 4",
            f"MOV R6, {boxes.base_addr}",
            "ADD R4, R4, R6",
            "LDG R20, [R4]",
            "MUL R4, R1, 4",
            "ADD R4, R4, 1",
            "MUL R4, R4, 4",
            f"MOV R6, {boxes.base_addr}",
            "ADD R4, R4, R6",
            "LDG R21, [R4]",
            "MUL R4, R1, 4",
            "ADD R4, R4, 2",
            "MUL R4, R4, 4",
            f"MOV R6, {boxes.base_addr}",
            "ADD R4, R4, R6",
            "LDG R22, [R4]",
            "MUL R4, R1, 4",
            "ADD R4, R4, 3",
            "MUL R4, R4, 4",
            f"MOV R6, {boxes.base_addr}",
            "ADD R4, R4, R6",
            "LDG R23, [R4]",
            "MUL R4, R1, 4",
            "ADD R4, R4, 0",
            "MUL R4, R4, 4",
            f"MOV R6, {deltas.base_addr}",
            "ADD R4, R4, R6",
            "LDG R24, [R4]",
            "MUL R4, R1, 4",
            "ADD R4, R4, 1",
            "MUL R4, R4, 4",
            f"MOV R6, {deltas.base_addr}",
            "ADD R4, R4, R6",
            "LDG R25, [R4]",
            "MUL R4, R1, 4",
            "ADD R4, R4, 2",
            "MUL R4, R4, 4",
            f"MOV R6, {deltas.base_addr}",
            "ADD R4, R4, R6",
            "LDG R26, [R4]",
            "MUL R4, R1, 4",
            "ADD R4, R4, 3",
            "MUL R4, R4, 4",
            f"MOV R6, {deltas.base_addr}",
            "ADD R4, R4, R6",
            "LDG R27, [R4]",
        ]
    )

    # Decode components: cx, cy, w, h.
    # cx = box0 + delta0 * box2; cy = box1 + delta1 * box3.
    lines.extend(
        [
            "MUL R4, R24, R22",
            "ADD R4, R20, R4",
            "MUL R5, R25, R23",
            "ADD R5, R21, R5",
            "MUL R6, R26, 1.4426950408889634",
            "EXP2 R6, R6",
            "MUL R7, R22, R6",
            "MUL R8, R27, 1.4426950408889634",
            "EXP2 R8, R8",
            "MUL R9, R23, R8",
        ]
    )

    # Select based on coordinate.
    lines.extend(
        [
            "MUL R10, R7, 0.5",
            "SUB R28, R4, R10",
            "ADD R29, R4, R10",
            "MUL R11, R9, 0.5",
            "SUB R30, R5, R11",
            "ADD R31, R5, R11",
            "MOV R27, 0",
            "CMP.EQ R12, R3, 0",
            "MUL R13, R12, R28",
            "ADD R27, R27, R13",
            "CMP.EQ R14, R3, 1",
            "MUL R15, R14, R30",
            "ADD R27, R27, R15",
            "CMP.EQ R16, R3, 2",
            "MUL R17, R16, R29",
            "ADD R27, R27, R17",
            "CMP.EQ R18, R3, 3",
            "MUL R19, R18, R31",
            "ADD R27, R27, R19",
            "MUL R20, R0, 4",
            f"MOV R21, {output.base_addr}",
            "ADD R20, R20, R21",
            "STG [R20], R27",
        ]
    )
    lines.extend(_finish())
    return "\n".join(lines) + "\n", _launch_1d(total)


def _nms_isa(
    node: Any, layout: dict[str, Any]
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    boxes = layout[node.inputs[0].name]
    scores = layout[node.inputs[1].name]
    output = layout[node.outputs[0].name]
    if len(boxes.shape) != 2 or boxes.shape[1] != 4:
        raise NotImplementedError("nms lowering expects (N,4) boxes")
    num_boxes = boxes.shape[0]
    threshold = float(node.attrs.get("iou_threshold", 0.5))

    lines = _prologue(num_boxes)
    lines.append("MOV R31, R0")

    # Load this box's score and coordinates into persistent registers.
    lines.extend(
        [
            "MUL R4, R0, 4",
            f"MOV R5, {scores.base_addr}",
            "ADD R4, R4, R5",
            "LDG R20, [R4]",
            "MUL R4, R0, 4",
            "ADD R4, R4, 0",
            "MUL R4, R4, 4",
            f"MOV R5, {boxes.base_addr}",
            "ADD R4, R4, R5",
            "LDG R21, [R4]",
            "MUL R4, R0, 4",
            "ADD R4, R4, 1",
            "MUL R4, R4, 4",
            f"MOV R5, {boxes.base_addr}",
            "ADD R4, R4, R5",
            "LDG R22, [R4]",
            "MUL R4, R0, 4",
            "ADD R4, R4, 2",
            "MUL R4, R4, 4",
            f"MOV R5, {boxes.base_addr}",
            "ADD R4, R4, R5",
            "LDG R23, [R4]",
            "MUL R4, R0, 4",
            "ADD R4, R4, 3",
            "MUL R4, R4, 4",
            f"MOV R5, {boxes.base_addr}",
            "ADD R4, R4, R5",
            "LDG R24, [R4]",
            "MOV R30, 1",
        ]
    )

    for j in range(num_boxes):
        lines.extend(
            _nms_pair(j, num_boxes, threshold, scores.base_addr, boxes.base_addr)
        )

    lines.extend(
        [
            "MOV R4, R31",
            "MOV R5, -1",
            "SEL R6, R30, R4, R5",
            "MUL R7, R0, 4",
            f"MOV R8, {output.base_addr}",
            "ADD R7, R7, R8",
            "STG [R7], R6",
        ]
    )
    lines.extend(_finish())
    return "\n".join(lines) + "\n", _launch_1d(num_boxes)


def _nms_pair(
    j: int,
    num_boxes: int,
    threshold: float,
    scores_base: int,
    boxes_base: int,
) -> list[str]:
    del num_boxes
    return [
        f"MOV R4, {j}",
        "MUL R4, R4, 4",
        f"MOV R5, {scores_base}",
        "ADD R4, R4, R5",
        "LDG R6, [R4]",
        "CMP.GT R7, R6, R20",
        f"MOV R4, {j * 4 + 0}",
        "MUL R4, R4, 4",
        f"MOV R5, {boxes_base}",
        "ADD R4, R4, R5",
        "LDG R8, [R4]",
        f"MOV R4, {j * 4 + 1}",
        "MUL R4, R4, 4",
        f"MOV R5, {boxes_base}",
        "ADD R4, R4, R5",
        "LDG R9, [R4]",
        f"MOV R4, {j * 4 + 2}",
        "MUL R4, R4, 4",
        f"MOV R5, {boxes_base}",
        "ADD R4, R4, R5",
        "LDG R10, [R4]",
        f"MOV R4, {j * 4 + 3}",
        "MUL R4, R4, 4",
        f"MOV R5, {boxes_base}",
        "ADD R4, R4, R5",
        "LDG R11, [R4]",
        # inter coords
        "MAX R12, R21, R8",
        "MAX R13, R22, R9",
        "MIN R14, R23, R10",
        "MIN R15, R24, R11",
        "SUB R16, R14, R12",
        "MOV R17, 0",
        "MAX R16, R16, R17",
        "SUB R18, R15, R13",
        "MAX R18, R18, R17",
        "MUL R16, R16, R18",
        # area_i
        "SUB R4, R23, R21",
        "SUB R5, R24, R22",
        "MUL R6, R4, R5",
        # area_j
        "SUB R4, R10, R8",
        "SUB R5, R11, R9",
        "MUL R4, R4, R5",
        # union = area_i + area_j - inter
        "ADD R5, R6, R4",
        "SUB R5, R5, R16",
        "MOV R8, 1",
        "MAX R5, R5, R8",
        "DIV R9, R16, R5",
        f"CMP.GT R10, R9, {threshold}",
        "MUL R11, R7, R10",
        "MOV R12, 1",
        "SUB R12, R12, R11",
        "MUL R30, R30, R12",
    ]
