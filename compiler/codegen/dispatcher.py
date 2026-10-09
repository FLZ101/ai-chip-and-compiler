"""Unified C++ ``mgpu-opt`` lowering dispatcher."""

from __future__ import annotations

from typing import Any

from ..model_ir import Graph, Op

CPP_SUPPORTED_OPS = {
    Op.ADD,
    Op.SUB,
    Op.MUL,
    Op.DIV,
    Op.REDUCE_SUM,
    Op.REDUCE_MEAN,
    Op.REDUCE_MAX,
    Op.MATMUL,
    Op.RELU,
    Op.LEAKY_RELU,
    Op.SILU,
    Op.GELU,
    Op.SIGMOID,
    Op.CONV2D,
    Op.DEPTHWISE_CONV2D,
    Op.MAX_POOL2D,
    Op.AVG_POOL2D,
    Op.GLOBAL_AVG_POOL,
    Op.ROPE,
    Op.CAUSAL_MASK,
    Op.KV_CACHE_APPEND,
    Op.KV_CACHE_GATHER,
    Op.RESHAPE,
    Op.TRANSPOSE,
    Op.CONCAT,
    Op.SLICE,
    Op.PAD,
    Op.RESIZE,
    Op.BATCH_NORM,
    Op.GROUP_NORM,
    Op.BOX_DECODE,
    Op.NMS,
    Op.SOFTMAX,
    Op.LAYER_NORM,
    Op.RMS_NORM,
    Op.SCALED_DOT_PRODUCT_ATTENTION,
    Op.MASKED_ATTENTION,
}


def lower_kernel(
    graph: Graph,
    mlir_text: str,
    layout: dict[str, Any],
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Lower one supported Model IR node to simulator ISA through C++."""
    if len(graph.nodes) != 1 or len(graph.outputs) != 1:
        raise NotImplementedError("lower_kernel supports one single-output node")

    node = graph.nodes[0]
    if node.op not in CPP_SUPPORTED_OPS:
        raise NotImplementedError(f"no C++ lowering for op {node.op.value!r}")

    from ..mlir.mgpu_runner import lower_with_mgpu_opt

    return lower_with_mgpu_opt(graph, mlir_text, layout)


def lower_common_kernel(
    graph: Graph,
    mlir_text: str,
    layout: dict[str, Any],
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Backward-compatible alias for tests that imported common kernels."""
    return lower_kernel(graph, mlir_text, layout)
