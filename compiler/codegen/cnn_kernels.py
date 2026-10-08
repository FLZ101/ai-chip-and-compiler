"""ISA code generation for ordinary CNN kernels."""

from __future__ import annotations

import math
from typing import Any

from ..model_ir import Graph, Op


def lower_cnn_kernel(
    graph: Graph,
    mlir_text: str,
    layout: dict[str, Any],
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Lower one CNN operator to simulator ISA."""
    del mlir_text

    if len(graph.nodes) != 1 or len(graph.outputs) != 1:
        raise NotImplementedError("CNN lowering supports one output")

    node = graph.nodes[0]
    if node.op == Op.CONV2D:
        return _conv2d_isa(node, layout)
    if node.op == Op.DEPTHWISE_CONV2D:
        return _depthwise_conv2d_isa(node, layout)
    if node.op == Op.MAX_POOL2D:
        return _pool2d_isa(node, layout, "MAX")
    if node.op == Op.AVG_POOL2D:
        return _pool2d_isa(node, layout, "ADD")
    if node.op == Op.GLOBAL_AVG_POOL:
        return _global_avg_pool_isa(node, layout)
    raise NotImplementedError(f"no CNN lowering for op {node.op.value!r}")


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


def _mma_lane_index() -> list[str]:
    return [
        "IDIV R24, tid.x, 4",
        "MUL R25, R24, 4",
        "SUB R25, tid.x, R25",
    ]


def _conv2d_mma_isa(
    image: Any,
    weight: Any,
    output: Any,
    bias: Any | None,
    *,
    c: int,
    h: int,
    w: int,
    f: int,
    kh: int,
    kw: int,
    oh: int,
    ow: int,
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Lower one 2D conv to a padded 16x16x16 MMA.

    The convolution is viewed as ``im2col(image) @ weight_matrix``.  A is the
    im2col patch matrix with ``spatial_size`` rows and ``c*kh*kw`` reduction
    columns.  B is the reshaped weight matrix with reduction rows and output
    channels as columns.  Invalid rows/columns are masked to zero.
    """
    reduction_size = c * kh * kw
    spatial_size = oh * ow
    stride_y, stride_x = stride
    pad_y, pad_x = padding

    lines = _mma_lane_index()
    for register in range(16, 24):
        lines.append(f"MOV R{register}, 0")

    for slot in range(8):
        row_group = slot // 4
        col_group = slot % 4
        lines.extend(
            _conv2d_a_fragment(
                slot,
                row_group=row_group,
                col_group=col_group,
                image=image,
                c=c,
                h=h,
                w=w,
                kh=kh,
                kw=kw,
                ow=ow,
                spatial_size=spatial_size,
                reduction_size=reduction_size,
                stride_y=stride_y,
                stride_x=stride_x,
                pad_y=pad_y,
                pad_x=pad_x,
            )
        )
        lines.extend(
            _conv2d_b_fragment(
                8 + slot,
                row_group=row_group,
                col_group=col_group,
                weight=weight,
                c=c,
                kh=kh,
                kw=kw,
                f=f,
                reduction_size=reduction_size,
            )
        )

    lines.append("MMA R16, R0, R8, R16")

    for slot in range(8):
        row_group = slot // 4
        col_group = slot % 4
        lines.extend(
            _store_mma_fragment_to_shared(
                16 + slot, row_group=row_group, col_group=col_group
            )
        )
    lines.append("BAR.SYNC")

    # One SIMT copy phase writes the valid padded C matrix to the real NCHW
    # output.  The shared tile is padded to 16 columns; only spatial_size rows
    # and f columns are real.
    lines.extend(_prologue(spatial_size * f))
    lines.extend(
        [
            f"IDIV R1, R0, {spatial_size}",
            f"MUL R2, R1, {spatial_size}",
            "SUB R3, R0, R2",
            f"IDIV R4, R3, {ow}",
            f"MUL R5, R4, {ow}",
            "SUB R6, R3, R5",
            "MUL R7, R3, 16",
            "ADD R7, R7, R1",
            "MUL R7, R7, 4",
            "LDS R8, [R7]",
        ]
    )
    if bias is not None:
        lines.extend(
            [
                "MUL R9, R1, 4",
                f"MOV R10, {bias.base_addr}",
                "ADD R9, R9, R10",
                "LDG R11, [R9]",
                "ADD R8, R8, R11",
            ]
        )
    lines.extend(
        [
            f"MUL R9, R1, {oh}",
            "ADD R9, R9, R4",
            f"MUL R9, R9, {ow}",
            "ADD R9, R9, R6",
            "MUL R9, R9, 4",
            f"MOV R10, {output.base_addr}",
            "ADD R9, R9, R10",
            "STG [R9], R8",
        ]
    )
    lines.extend(_finish())
    return "\n".join(lines) + "\n", ((1,), (32,))


def _conv2d_a_fragment(
    dest: int,
    *,
    row_group: int,
    col_group: int,
    image: Any,
    c: int,
    h: int,
    w: int,
    kh: int,
    kw: int,
    ow: int,
    spatial_size: int,
    reduction_size: int,
    stride_y: int,
    stride_x: int,
    pad_y: int,
    pad_x: int,
) -> list[str]:
    area = kh * kw
    return [
        f"MOV R26, {row_group * 8}",
        "ADD R26, R26, R24",
        f"MOV R27, {col_group * 4}",
        "ADD R27, R27, R25",
        f"MOV R39, {spatial_size}",
        "CMP.LT R28, R26, R39",
        f"MOV R39, {reduction_size}",
        "CMP.LT R40, R27, R39",
        "MUL R28, R28, R40",
        f"MOV R39, {spatial_size - 1}",
        "MIN R29, R26, R39",
        f"MOV R39, {reduction_size - 1}",
        "MIN R30, R27, R39",
        f"IDIV R31, R29, {ow}",
        f"MUL R32, R31, {ow}",
        "SUB R32, R29, R32",
        f"IDIV R33, R30, {area}",
        f"MUL R34, R33, {area}",
        "SUB R34, R30, R34",
        f"IDIV R35, R34, {kw}",
        f"MUL R36, R35, {kw}",
        "SUB R36, R34, R36",
        f"MOV R37, {stride_y}",
        "MUL R37, R31, R37",
        "ADD R37, R37, R35",
        f"SUB R37, R37, {pad_y}",
        f"MOV R38, {stride_x}",
        "MUL R38, R32, R38",
        "ADD R38, R38, R36",
        f"SUB R38, R38, {pad_x}",
        "MOV R39, 0",
        "MAX R37, R37, R39",
        f"MOV R39, {h - 1}",
        "MIN R37, R37, R39",
        "MOV R39, 0",
        "MAX R38, R38, R39",
        f"MOV R39, {w - 1}",
        "MIN R38, R38, R39",
        f"MOV R41, {h}",
        "MUL R41, R33, R41",
        "ADD R41, R41, R37",
        f"MUL R41, R41, {w}",
        "ADD R41, R41, R38",
        "MUL R41, R41, 4",
        f"ADD R41, R41, {image.base_addr}",
        "LDG R42, [R41]",
        "MOV R39, 0",
        f"SEL R{dest}, R28, R42, R39",
    ]


def _conv2d_b_fragment(
    dest: int,
    *,
    row_group: int,
    col_group: int,
    weight: Any,
    c: int,
    kh: int,
    kw: int,
    f: int,
    reduction_size: int,
) -> list[str]:
    area = kh * kw
    return [
        f"MOV R26, {row_group * 8}",
        "ADD R26, R26, R24",
        f"MOV R27, {col_group * 4}",
        "ADD R27, R27, R25",
        f"MOV R39, {reduction_size}",
        "CMP.LT R28, R26, R39",
        f"MOV R39, {f}",
        "CMP.LT R40, R27, R39",
        "MUL R28, R28, R40",
        f"MOV R39, {reduction_size - 1}",
        "MIN R29, R26, R39",
        f"MOV R39, {f - 1}",
        "MIN R30, R27, R39",
        f"IDIV R33, R29, {area}",
        f"MUL R34, R33, {area}",
        "SUB R34, R29, R34",
        f"IDIV R35, R34, {kw}",
        f"MUL R36, R35, {kw}",
        "SUB R36, R34, R36",
        f"MOV R37, {c}",
        "MUL R37, R30, R37",
        "ADD R37, R37, R33",
        f"MUL R37, R37, {kh}",
        "ADD R37, R37, R35",
        f"MUL R37, R37, {kw}",
        "ADD R37, R37, R36",
        "MUL R37, R37, 4",
        f"ADD R37, R37, {weight.base_addr}",
        "LDG R42, [R37]",
        "MOV R39, 0",
        f"SEL R{dest}, R28, R42, R39",
    ]


def _store_mma_fragment_to_shared(
    source: int, *, row_group: int, col_group: int
) -> list[str]:
    return [
        f"MOV R26, {row_group * 8}",
        "ADD R26, R26, R24",
        f"MOV R27, {col_group * 4}",
        "ADD R27, R27, R25",
        "MUL R28, R26, 16",
        "ADD R28, R28, R27",
        "MUL R28, R28, 4",
        f"STS [R28], R{source}",
    ]


def _conv2d_isa(
    node: Any, layout: dict[str, Any]
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    image = layout[node.inputs[0].name]
    weight = layout[node.inputs[1].name]
    output = layout[node.outputs[0].name]
    bias = layout[node.inputs[2].name] if len(node.inputs) == 3 else None

    if image.shape[0] != 1:
        raise NotImplementedError("conv2d lowering supports batch size 1")
    if int(node.attrs.get("groups", 1)) != 1:
        raise NotImplementedError("conv2d lowering supports groups=1")
    dilation = _pair(node.attrs.get("dilation", 1), "dilation")
    if dilation != (1, 1):
        raise NotImplementedError("conv2d lowering supports dilation=1")

    n, c, h, w = image.shape
    f, c_weight, kh, kw = weight.shape
    if c != c_weight:
        raise NotImplementedError("conv2d input channels do not match weight")
    _, f_out, oh, ow = output.shape
    if f != f_out:
        raise NotImplementedError("conv2d output channels do not match weight")

    stride = _pair(node.attrs.get("strides", 1), "strides")
    padding = _pair(node.attrs.get("padding", 0), "padding")
    out_num = f * oh * ow

    reduction_size = c * kh * kw
    spatial_size = oh * ow
    if f <= 16 and reduction_size <= 16 and spatial_size <= 16:
        return _conv2d_mma_isa(
            image,
            weight,
            output,
            bias,
            c=c,
            h=h,
            w=w,
            f=f,
            kh=kh,
            kw=kw,
            oh=oh,
            ow=ow,
            stride=stride,
            padding=padding,
        )

    lines = _prologue(out_num)
    lines.extend(
        [
            f"IDIV R1, R0, {oh * ow}",
            f"MUL R4, R1, {oh * ow}",
            "SUB R5, R0, R4",
            f"IDIV R2, R5, {ow}",
            f"MUL R6, R2, {ow}",
            "SUB R3, R5, R6",
            "MOV R20, 0",
        ]
    )

    for ci in range(c):
        for ky in range(kh):
            for kx in range(kw):
                lines.extend(
                    _conv_tap(
                        image,
                        weight,
                        c_channels=c,
                        ci=ci,
                        ky=ky,
                        kx=kx,
                        h=h,
                        w=w,
                        kh=kh,
                        kw=kw,
                        stride=stride,
                        padding=padding,
                    )
                )

    if bias is not None:
        lines.extend(
            [
                "MUL R4, R1, 4",
                f"MOV R5, {bias.base_addr}",
                "ADD R4, R4, R5",
                "LDG R6, [R4]",
                "ADD R20, R20, R6",
            ]
        )

    lines.extend(
        [
            f"MUL R4, R1, {oh * ow}",
            f"MUL R5, R2, {ow}",
            "ADD R4, R4, R5",
            "ADD R4, R4, R3",
            "MUL R4, R4, 4",
            f"MOV R5, {output.base_addr}",
            "ADD R4, R4, R5",
            "STG [R4], R20",
        ]
    )
    lines.extend(_finish())
    return "\n".join(lines) + "\n", _launch_1d(out_num)


def _conv_tap(
    image: Any,
    weight: Any,
    *,
    c_channels: int,
    ci: int,
    ky: int,
    kx: int,
    h: int,
    w: int,
    kh: int,
    kw: int,
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> list[str]:
    ih_const = ky - padding[0]
    iw_const = kx - padding[1]
    lines = [
        f"MOV R4, {stride[0]}",
        "MUL R4, R2, R4",
        f"ADD R4, R4, {ih_const}",
        "CMP.GE R5, R4, 0",
        f"MOV R6, {h}",
        "CMP.LT R7, R4, R6",
        "MUL R8, R5, R7",
        "MOV R9, 0",
        "MAX R10, R4, R9",
        f"MOV R9, {h - 1}",
        "MIN R10, R10, R9",
        f"MOV R11, {stride[1]}",
        "MUL R11, R3, R11",
        f"ADD R11, R11, {iw_const}",
        "CMP.GE R12, R11, 0",
        f"MOV R13, {w}",
        "CMP.LT R14, R11, R13",
        "MUL R15, R12, R14",
        "MUL R18, R8, R15",
        "MOV R16, 0",
        "MAX R17, R11, R16",
        f"MOV R16, {w - 1}",
        "MIN R17, R17, R16",
        f"MOV R19, {ci * h}",
        "ADD R19, R19, R10",
        f"MUL R19, R19, {w}",
        "ADD R19, R19, R17",
        "MUL R19, R19, 4",
        f"MOV R4, {image.base_addr}",
        "ADD R19, R19, R4",
        "LDG R6, [R19]",
        "MOV R7, 0",
        "SEL R8, R18, R6, R7",
        f"MOV R9, {c_channels}",
        "MUL R9, R1, R9",
        f"ADD R9, R9, {ci}",
        f"MUL R9, R9, {kh}",
        f"ADD R9, R9, {ky}",
        f"MUL R9, R9, {kw}",
        f"ADD R9, R9, {kx}",
        "MUL R9, R9, 4",
        f"MOV R10, {weight.base_addr}",
        "ADD R9, R9, R10",
        "LDG R11, [R9]",
        "FMA R20, R8, R11, R20",
    ]
    return lines


def _depthwise_conv2d_isa(
    node: Any, layout: dict[str, Any]
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    image = layout[node.inputs[0].name]
    weight = layout[node.inputs[1].name]
    output = layout[node.outputs[0].name]
    bias = layout[node.inputs[2].name] if len(node.inputs) == 3 else None

    if image.shape[0] != 1:
        raise NotImplementedError("depthwise conv supports batch size 1")
    n, c, h, w = image.shape
    c_w, _, kh, kw = weight.shape
    if c != c_w:
        raise NotImplementedError("depthwise conv channel counts do not match")
    _, c_out, oh, ow = output.shape
    if c != c_out:
        raise NotImplementedError("depthwise conv output channels do not match")

    stride = _pair(node.attrs.get("strides", 1), "strides")
    padding = _pair(node.attrs.get("padding", 0), "padding")
    out_num = c * oh * ow

    lines = _prologue(out_num)
    lines.extend(
        [
            f"IDIV R1, R0, {oh * ow}",
            f"MUL R4, R1, {oh * ow}",
            "SUB R5, R0, R4",
            f"IDIV R2, R5, {ow}",
            f"MUL R6, R2, {ow}",
            "SUB R3, R5, R6",
            "MOV R20, 0",
        ]
    )

    for ky in range(kh):
        for kx in range(kw):
            lines.extend(
                _depthwise_tap(
                    image,
                    weight,
                    ky=ky,
                    kx=kx,
                    h=h,
                    w=w,
                    kh=kh,
                    kw=kw,
                    stride=stride,
                    padding=padding,
                )
            )

    if bias is not None:
        lines.extend(
            [
                "MUL R4, R1, 4",
                f"MOV R5, {bias.base_addr}",
                "ADD R4, R4, R5",
                "LDG R6, [R4]",
                "ADD R20, R20, R6",
            ]
        )

    lines.extend(
        [
            f"MUL R4, R1, {oh * ow}",
            f"MUL R5, R2, {ow}",
            "ADD R4, R4, R5",
            "ADD R4, R4, R3",
            "MUL R4, R4, 4",
            f"MOV R5, {output.base_addr}",
            "ADD R4, R4, R5",
            "STG [R4], R20",
        ]
    )
    lines.extend(_finish())
    return "\n".join(lines) + "\n", _launch_1d(out_num)


def _depthwise_tap(
    image: Any,
    weight: Any,
    *,
    ky: int,
    kx: int,
    h: int,
    w: int,
    kh: int,
    kw: int,
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> list[str]:
    # Depthwise weight layout is (C, 1, KH, KW), image layout is NCHW.
    # Channel is R1, output row is R2, output column is R3.
    ih_const = ky - padding[0]
    iw_const = kx - padding[1]
    lines = [
        f"MOV R4, {stride[0]}",
        "MUL R4, R2, R4",
        f"ADD R4, R4, {ih_const}",
        "CMP.GE R5, R4, 0",
        f"MOV R6, {h}",
        "CMP.LT R7, R4, R6",
        "MUL R8, R5, R7",
        "MOV R9, 0",
        "MAX R10, R4, R9",
        f"MOV R9, {h - 1}",
        "MIN R10, R10, R9",
        f"MOV R11, {stride[1]}",
        "MUL R11, R3, R11",
        f"ADD R11, R11, {iw_const}",
        "CMP.GE R12, R11, 0",
        f"MOV R13, {w}",
        "CMP.LT R14, R11, R13",
        "MUL R15, R12, R14",
        "MUL R18, R8, R15",
        "MOV R16, 0",
        "MAX R17, R11, R16",
        f"MOV R16, {w - 1}",
        "MIN R17, R17, R16",
        # Input address: (R1 * h + safe_h) * w + safe_w.
        f"MOV R19, {h}",
        "MUL R19, R1, R19",
        "ADD R19, R19, R10",
        f"MUL R19, R19, {w}",
        "ADD R19, R19, R17",
        "MUL R19, R19, 4",
        f"MOV R4, {image.base_addr}",
        "ADD R19, R19, R4",
        "LDG R6, [R19]",
        "MOV R7, 0",
        "SEL R8, R18, R6, R7",
        # Weight address: (R1 * kh + ky) * kw + kx.
        f"MOV R9, {kh}",
        "MUL R9, R1, R9",
        f"ADD R9, R9, {ky}",
        f"MUL R9, R9, {kw}",
        f"ADD R9, R9, {kx}",
        "MUL R9, R9, 4",
        f"MOV R10, {weight.base_addr}",
        "ADD R9, R9, R10",
        "LDG R11, [R9]",
        "FMA R20, R8, R11, R20",
    ]
    return lines


def _pool2d_isa(
    node: Any, layout: dict[str, Any], arith: str
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    image = layout[node.inputs[0].name]
    output = layout[node.outputs[0].name]
    if image.shape[0] != 1:
        raise NotImplementedError("pool lowering supports batch size 1")
    if int(node.attrs.get("padding", 0)) != 0:
        raise NotImplementedError("pool lowering supports padding=0 only")

    n, c, h, w = image.shape
    _, c_out, oh, ow = output.shape
    if c != c_out:
        raise NotImplementedError("pool output channels do not match input")
    kernel = _pair(node.attrs.get("kernel", 2), "kernel")
    stride = _pair(node.attrs.get("strides", kernel), "strides")
    out_num = c * oh * ow

    identity = -1.0e30 if arith == "MAX" else 0.0
    lines = _prologue(out_num)
    lines.extend(
        [
            f"IDIV R1, R0, {oh * ow}",
            f"MUL R4, R1, {oh * ow}",
            "SUB R5, R0, R4",
            f"IDIV R2, R5, {ow}",
            f"MUL R6, R2, {ow}",
            "SUB R3, R5, R6",
            f"MOV R20, {identity}",
        ]
    )

    for ky in range(kernel[0]):
        for kx in range(kernel[1]):
            ih = ky
            iw = kx
            lines.extend(
                [
                    f"MOV R4, {stride[0]}",
                    "MUL R4, R2, R4",
                    f"ADD R4, R4, {ih}",
                    f"MOV R5, {stride[1]}",
                    "MUL R5, R3, R5",
                    f"ADD R5, R5, {iw}",
                    f"MOV R6, {h}",
                    "MUL R6, R1, R6",
                    "ADD R6, R6, R4",
                    f"MUL R6, R6, {w}",
                    "ADD R6, R6, R5",
                    "MUL R6, R6, 4",
                    f"MOV R7, {image.base_addr}",
                    "ADD R6, R6, R7",
                    "LDG R8, [R6]",
                    f"{arith} R20, R20, R8",
                ]
            )

    if arith == "ADD":
        area = kernel[0] * kernel[1]
        lines.append(f"DIV R20, R20, {area}")

    lines.extend(
        [
            f"MUL R4, R1, {oh * ow}",
            f"MUL R5, R2, {ow}",
            "ADD R4, R4, R5",
            "ADD R4, R4, R3",
            "MUL R4, R4, 4",
            f"MOV R5, {output.base_addr}",
            "ADD R4, R4, R5",
            "STG [R4], R20",
        ]
    )
    lines.extend(_finish())
    return "\n".join(lines) + "\n", _launch_1d(out_num)


def _global_avg_pool_isa(
    node: Any, layout: dict[str, Any]
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    image = layout[node.inputs[0].name]
    output = layout[node.outputs[0].name]
    if image.shape[0] != 1:
        raise NotImplementedError("global avg pool supports batch size 1")
    n, c, h, w = image.shape
    _, c_out, oh, ow = output.shape
    if c != c_out or (oh, ow) != (1, 1):
        raise NotImplementedError("global avg pool output must be NCHW with H=W=1")

    out_num = c
    lines = _prologue(out_num)
    lines.append("MOV R1, R0")
    lines.append("MOV R20, 0")
    for ih in range(h):
        for iw in range(w):
            lines.extend(
                [
                    f"MOV R4, {h}",
                    "MUL R4, R1, R4",
                    f"ADD R4, R4, {ih}",
                    f"MUL R4, R4, {w}",
                    f"ADD R4, R4, {iw}",
                    "MUL R4, R4, 4",
                    f"MOV R5, {image.base_addr}",
                    "ADD R4, R4, R5",
                    "LDG R6, [R4]",
                    "ADD R20, R20, R6",
                ]
            )
    area = h * w
    lines.extend(
        [
            f"DIV R20, R20, {area}",
            "MUL R4, R1, 4",
            f"MOV R5, {output.base_addr}",
            "ADD R4, R4, R5",
            "STG [R4], R20",
        ]
    )
    lines.extend(_finish())
    return "\n".join(lines) + "\n", _launch_1d(out_num)


def _pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, tuple) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise NotImplementedError(f"{name} must be an int or pair")
