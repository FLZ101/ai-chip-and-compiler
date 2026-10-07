"""Unified dispatcher for all Model IR ISA lowerings."""

from __future__ import annotations

from typing import Any

from ..model_ir import Graph, Op
from .activation_kernels import lower_activation_kernel
from .common_kernels import lower_common_kernel
from .shape_kernels import lower_shape_kernel


def lower_kernel(
    graph: Graph,
    mlir_text: str,
    layout: dict[str, Any],
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Lower one supported Model IR node to simulator ISA."""
    if len(graph.nodes) != 1 or len(graph.outputs) != 1:
        raise NotImplementedError("lower_kernel supports one single-output node")

    node = graph.nodes[0]
    if node.op in {
        Op.RELU,
        Op.LEAKY_RELU,
        Op.SILU,
        Op.GELU,
        Op.SIGMOID,
    }:
        return lower_activation_kernel(graph, mlir_text, layout)
    if node.op in {
        Op.RESHAPE,
        Op.TRANSPOSE,
        Op.CONCAT,
        Op.SLICE,
        Op.PAD,
        Op.RESIZE,
    }:
        return lower_shape_kernel(graph, mlir_text, layout)
    return lower_common_kernel(graph, mlir_text, layout)
