"""Unified dispatcher for all Model IR ISA lowerings."""

from __future__ import annotations

from typing import Any

from ..model_ir import Graph, Op
from .activation_kernels import lower_activation_kernel
from .attention_kernels import lower_attention_kernel
from .cnn_kernels import lower_cnn_kernel
from .common_kernels import lower_common_kernel
from .norm_detection_kernels import lower_norm_detection_kernel
from .shape_kernels import lower_shape_kernel
from .transformer_kernels import lower_transformer_kernel

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
    """Lower one supported Model IR node to simulator ISA."""
    if len(graph.nodes) != 1 or len(graph.outputs) != 1:
        raise NotImplementedError("lower_kernel supports one single-output node")

    node = graph.nodes[0]
    if node.op in CPP_SUPPORTED_OPS:
        from ..mlir.mgpu_runner import MGpuOptError, lower_with_mgpu_opt

        if node.op == Op.CONV2D and len(node.inputs) != 2:
            return lower_kernel_python(graph, mlir_text, layout)
        try:
            return lower_with_mgpu_opt(graph, mlir_text, layout)
        except MGpuOptError:
            raise

    return lower_kernel_python(graph, mlir_text, layout)


def lower_kernel_python(
    graph: Graph,
    mlir_text: str,
    layout: dict[str, Any],
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Python reference backend used for unsupported or multi-node cases."""
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
    if node.op in {
        Op.CONV2D,
        Op.DEPTHWISE_CONV2D,
        Op.MAX_POOL2D,
        Op.AVG_POOL2D,
        Op.GLOBAL_AVG_POOL,
    }:
        return lower_cnn_kernel(graph, mlir_text, layout)
    if node.op in {
        Op.BATCH_NORM,
        Op.GROUP_NORM,
        Op.BOX_DECODE,
        Op.NMS,
    }:
        return lower_norm_detection_kernel(graph, mlir_text, layout)
    if node.op in {
        Op.SOFTMAX,
        Op.LAYER_NORM,
        Op.RMS_NORM,
    }:
        return lower_transformer_kernel(graph, mlir_text, layout)
    if node.op in {
        Op.ROPE,
        Op.CAUSAL_MASK,
        Op.SCALED_DOT_PRODUCT_ATTENTION,
        Op.MASKED_ATTENTION,
        Op.KV_CACHE_APPEND,
        Op.KV_CACHE_GATHER,
    }:
        return lower_attention_kernel(graph, mlir_text, layout)
    return lower_common_kernel(graph, mlir_text, layout)
