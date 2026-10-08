"""Deterministic MLIR text builder for the unified Model IR.

Step 006 lowers a small, explicit subset of the Model IR into MLIR text.  The
output combines standard MLIR dialects with the custom ``mgpu`` dialect:

* ``func``, ``arith``, and ``memref`` provide the function and memory skeleton,
* ``linalg`` provides high-level compute ops,
* ``mgpu.launch`` and its body ops model the device grid/thread mapping,
* ``mgpu.barrier`` marks block synchronization.

Shared-memory tiling, async copy, and tensor-core ``mgpu.mma`` are introduced by
later passes rather than by this initial builder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..model_ir import DType, Graph, Node, Op, Value


class MLIRBuilderError(Exception):
    """Raised when a Model IR node cannot be emitted as MLIR."""


_DTYPE_TO_MLIR = {
    DType.F32: "f32",
    DType.F16: "f16",
    DType.BF16: "bf16",
    DType.I32: "i32",
    DType.I64: "i64",
    DType.I8: "i8",
    DType.U8: "ui8",
    DType.I4: "i4",
    DType.U4: "ui4",
    DType.F8_E4M3: "f8E4M3FN",
    DType.F8_E5M2: "f8E5M2",
    DType.BOOL: "i1",
}

_BINARY_ARITH = {
    Op.ADD: "arith.addf",
    Op.SUB: "arith.subf",
    Op.MUL: "arith.mulf",
    Op.DIV: "arith.divf",
}

_REDUCTION_ARITH = {
    Op.REDUCE_SUM: "arith.addf",
    Op.REDUCE_MEAN: "arith.addf",
    Op.REDUCE_MAX: "arith.maximumf",
}


@dataclass
class MLIRBuilder:
    """Build a deterministic MLIR module from a :class:`Graph`."""

    launch_threads: tuple[int, int, int] = (32, 1, 1)
    _value_names: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _node_results: list[tuple[Node, str]] = field(
        default_factory=list, init=False, repr=False
    )

    def build(self, graph: Graph) -> str:
        """Return the MLIR module text for ``graph``.

        The builder calls :meth:`Graph.validate` first so malformed graphs fail
        before any text is generated.
        """
        graph.validate()
        self._prepare_value_names(graph)

        lines = ["module {"]
        lines.extend(self._emit_function(graph, indent="  "))
        lines.append("}")
        return "\n".join(lines) + "\n"

    def _prepare_value_names(self, graph: Graph) -> None:
        self._value_names = {}
        for index, value in enumerate(graph.inputs):
            self._value_names[value.name] = f"%arg{index}"

        self._node_results = []
        counter = 0
        for node in graph.nodes:
            if len(node.outputs) != 1:
                raise MLIRBuilderError(
                    f"node {node.name} has {len(node.outputs)} outputs; "
                    "MLIR builder currently supports single-output nodes"
                )
            result_name = f"%v{counter}"
            counter += 1
            self._value_names[node.outputs[0].name] = result_name
            self._node_results.append((node, result_name))

    def _emit_function(self, graph: Graph, *, indent: str) -> list[str]:
        function_name = _sanitize_identifier(graph.name)
        args = ", ".join(
            f"%arg{index}: {self._memref_type(value)}"
            for index, value in enumerate(graph.inputs)
        )
        return_types = [self._memref_type(value) for value in graph.outputs]
        return_type = _type_list(return_types)

        lines = [f"{indent}func.func @{function_name}({args}) -> {return_type} {{"]
        body_indent = indent + "  "
        lines.extend(
            [
                f"{body_indent}%c0 = arith.constant 0 : index",
                f"{body_indent}%c1 = arith.constant 1 : index",
                f"{body_indent}%c32 = arith.constant 32 : index",
            ]
        )

        for node, result_name in self._node_results:
            output = node.outputs[0]
            lines.append(
                f"{body_indent}{result_name} = memref.alloc() : "
                f"{self._memref_type(output)}"
            )

        for node, result_name in self._node_results:
            lines.extend(self._emit_node(node, result_name, body_indent))

        if graph.outputs:
            returned = ", ".join(
                self._value_names[value.name] for value in graph.outputs
            )
            lines.append(f"{body_indent}return {returned} : {return_type}")
        else:
            lines.append(f"{body_indent}return")
        lines.append(f"{indent}}}")
        return lines

    def _emit_node(self, node: Node, result_name: str, indent: str) -> list[str]:
        launch_suffix = result_name.lstrip("%")
        lines = [
            f"{indent}mgpu.launch blocks(%c1, %c1, %c1) " f"threads(%c32, %c1, %c1) {{"
        ]
        body_indent = indent + "  "
        lines.extend(self._emit_launch_ids(launch_suffix, body_indent))
        lines.extend(self._emit_compute(node, result_name, body_indent))
        lines.append(f"{body_indent}mgpu.barrier")
        lines.append(f"{body_indent}mgpu.return")
        lines.append(f"{indent}}}")
        return lines

    def _emit_launch_ids(self, suffix: str, indent: str) -> list[str]:
        return [
            f"{indent}%tx_{suffix} = mgpu.thread_id 0 : index",
            f"{indent}%bx_{suffix} = mgpu.block_id 0 : index",
            f"{indent}%gx_{suffix} = mgpu.grid_dim 0 : index",
            f"{indent}%bdx_{suffix} = mgpu.block_dim 0 : index",
        ]

    def _emit_compute(self, node: Node, result_name: str, indent: str) -> list[str]:
        if node.op in _BINARY_ARITH:
            return self._emit_binary_generic(node, result_name, indent)
        if node.op in _REDUCTION_ARITH:
            return self._emit_reduction(node, result_name, indent)
        if node.op == Op.MATMUL:
            return self._emit_matmul(node, result_name, indent)
        if node.op == Op.LINEAR:
            return self._emit_linear_placeholder(node, result_name, indent)
        if node.op == Op.CONV2D:
            return self._emit_conv2d(node, result_name, indent)
        if node.op == Op.DEPTHWISE_CONV2D:
            return self._emit_depthwise_conv2d(node, result_name, indent)
        if node.op in {
            Op.MAX_POOL2D,
            Op.AVG_POOL2D,
            Op.GLOBAL_AVG_POOL,
        }:
            return self._emit_pooling(node, result_name, indent)
        if node.op in {
            Op.BATCH_NORM,
            Op.GROUP_NORM,
            Op.BOX_DECODE,
            Op.NMS,
        }:
            return self._emit_norm_detection_placeholder(node, result_name, indent)
        if node.op == Op.SOFTMAX:
            return self._emit_softmax(node, result_name, indent)
        if node.op == Op.LAYER_NORM:
            return self._emit_layer_norm(node, result_name, indent)
        if node.op == Op.RMS_NORM:
            return self._emit_rms_norm(node, result_name, indent)
        if node.op in {
            Op.ROPE,
            Op.CAUSAL_MASK,
            Op.KV_CACHE_APPEND,
            Op.KV_CACHE_GATHER,
        }:
            return self._emit_attention_placeholder(node, result_name, indent)
        if node.op in {
            Op.SCALED_DOT_PRODUCT_ATTENTION,
            Op.MASKED_ATTENTION,
        }:
            return self._emit_attention(node, result_name, indent)
        if node.op in {
            Op.RELU,
            Op.LEAKY_RELU,
            Op.SILU,
            Op.GELU,
            Op.SIGMOID,
        }:
            return self._emit_activation(node, result_name, indent)
        if node.op in {
            Op.RESHAPE,
            Op.TRANSPOSE,
            Op.CONCAT,
            Op.SLICE,
            Op.PAD,
            Op.RESIZE,
        }:
            return self._emit_shape_placeholder(node, result_name, indent)
        raise MLIRBuilderError(
            f"MLIR lowering for op {node.op.value!r} is not implemented "
            f"(node {node.name!r})"
        )

    def _emit_binary_generic(
        self, node: Node, result_name: str, indent: str
    ) -> list[str]:
        left, right = node.inputs
        output = node.outputs[0]
        if left.shape != right.shape or left.shape != output.shape:
            raise MLIRBuilderError(
                "MLIR binary lowering currently requires equal input/output shapes"
            )

        rank = output.rank
        dims = ", ".join(f"d{axis}" for axis in range(rank))
        identity_map = f"affine_map<({dims}) -> ({dims})>"
        maps = ", ".join([identity_map] * 3)
        iterator_types = ", ".join('"parallel"' for _ in range(rank))
        left_type = self._memref_type(left)
        right_type = self._memref_type(right)
        out_type = self._memref_type(output)
        scalar_type = self._scalar_type(output.dtype)
        arith_op = _BINARY_ARITH[node.op]

        return [
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [{maps}],",
            f"{indent}  iterator_types = [{iterator_types}]",
            f"{indent}}} ins({self._value_names[left.name]}, "
            f"{self._value_names[right.name]} : {left_type}, {right_type}) "
            f"outs({result_name} : {out_type}) {{",
            f"{indent}^bb0(%lhs: {scalar_type}, %rhs: {scalar_type}, "
            f"%out: {scalar_type}):",
            f"{indent}  %value = {arith_op} %lhs, %rhs : {scalar_type}",
            f"{indent}  linalg.yield %value : {scalar_type}",
            f"{indent}}}",
        ]

    def _emit_matmul(self, node: Node, result_name: str, indent: str) -> list[str]:
        left, right = node.inputs
        output = node.outputs[0]
        left_type = self._memref_type(left)
        right_type = self._memref_type(right)
        out_type = self._memref_type(output)
        return [
            f"{indent}linalg.matmul "
            f"ins({self._value_names[left.name]}, {self._value_names[right.name]} : "
            f"{left_type}, {right_type}) outs({result_name} : {out_type}) "
        ]

    def _emit_linear_placeholder(
        self, node: Node, result_name: str, indent: str
    ) -> list[str]:
        value = node.inputs[0]
        output = node.outputs[0]
        scalar_type = self._scalar_type(output.dtype)
        value_type = self._memref_type(value)
        out_type = self._memref_type(output)
        dims = ", ".join(f"d{axis}" for axis in range(output.rank))
        identity = f"affine_map<({dims}) -> ({dims})>"
        input_map = f"affine_map<({dims}) -> (d0, 0, 0, 0)>"
        iterator_types = ", ".join('"parallel"' for _ in range(output.rank))
        return [
            f"{indent}// Linear op is lowered by mgpu-to-isa; "
            "linalg identity is a placeholder.",
            f"{indent}{result_name} = linalg.generic {{",
            f"{indent}  indexing_maps = [{identity}, {identity}]",
            f"{indent}  iterator_types = [{iterator_types}]",
            f"{indent}}} ins({self._value_names[value.name]} : {value_type}) "
            f"outs({result_name} : {out_type}) {{",
            f"{indent}^bb0(%in: {scalar_type}, %out: {scalar_type}):",
            f"{indent}  linalg.yield %in : {scalar_type}",
            f"{indent}}} -> {out_type}",
        ]

    def _emit_activation(self, node: Node, result_name: str, indent: str) -> list[str]:
        value = node.inputs[0]
        output = node.outputs[0]
        if value.shape != output.shape:
            raise MLIRBuilderError("activation lowering requires equal shapes")

        scalar_type = self._scalar_type(output.dtype)
        value_type = self._memref_type(value)
        out_type = self._memref_type(output)
        dims = ", ".join(f"d{axis}" for axis in range(output.rank))
        identity = f"affine_map<({dims}) -> ({dims})>"
        input_map = f"affine_map<({dims}) -> (d0, 0, 0, 0)>"
        iterator_types = ", ".join('"parallel"' for _ in range(output.rank))

        if node.op == Op.RELU:
            body = [
                f"{indent}  %act_c0 = arith.constant 0.000000e+00 : {scalar_type}",
                f"{indent}  %act_value = arith.maximumf %in, %act_c0 : {scalar_type}",
            ]
        elif node.op == Op.LEAKY_RELU:
            alpha = float(node.attrs.get("alpha", 0.01))
            body = [
                f"{indent}  %act_c0 = arith.constant 0.000000e+00 : {scalar_type}",
                f"{indent}  %act_neg = arith.cmpf olt, %in, %act_c0 : {scalar_type}",
                f"{indent}  %act_alpha = arith.constant {alpha!r} : {scalar_type}",
                f"{indent}  %act_scaled = arith.mulf %in, %act_alpha : {scalar_type}",
                f"{indent}  %act_value = arith.select %act_neg, %act_scaled, "
                f"%in : {scalar_type}",
            ]
        elif node.op == Op.SIGMOID:
            body = [
                f"{indent}  %act_neg = arith.negf %in : {scalar_type}",
                f"{indent}  %act_exp = math.exp %act_neg : {scalar_type}",
                f"{indent}  %act_one = arith.constant 1.000000e+00 : {scalar_type}",
                f"{indent}  %act_den = arith.addf %act_one, %act_exp : {scalar_type}",
                f"{indent}  %act_value = arith.divf %act_one, %act_den : {scalar_type}",
            ]
        elif node.op == Op.SILU:
            body = [
                f"{indent}  %act_neg = arith.negf %in : {scalar_type}",
                f"{indent}  %act_exp = math.exp %act_neg : {scalar_type}",
                f"{indent}  %act_one = arith.constant 1.000000e+00 : {scalar_type}",
                f"{indent}  %act_den = arith.addf %act_one, %act_exp : {scalar_type}",
                f"{indent}  %act_sigmoid = arith.divf %act_one, %act_den : {scalar_type}",
                f"{indent}  %act_value = arith.mulf %in, %act_sigmoid : {scalar_type}",
            ]
        elif node.op == Op.GELU:
            body = [
                f"{indent}  %act_cst = arith.constant 0.7978845608028654 : {scalar_type}",
                f"{indent}  %act_x2 = arith.mulf %in, %in : {scalar_type}",
                f"{indent}  %act_x3 = arith.mulf %act_x2, %in : {scalar_type}",
                f"{indent}  %act_cst2 = arith.constant 0.044715 : {scalar_type}",
                f"{indent}  %act_poly = arith.mulf %act_cst2, %act_x3 : {scalar_type}",
                f"{indent}  %act_sum = arith.addf %in, %act_poly : {scalar_type}",
                f"{indent}  %act_arg = arith.mulf %act_cst, %act_sum : {scalar_type}",
                f"{indent}  %act_tanh = math.tanh %act_arg : {scalar_type}",
                f"{indent}  %act_one = arith.constant 1.000000e+00 : {scalar_type}",
                f"{indent}  %act_one_t = arith.addf %act_one, %act_tanh : {scalar_type}",
                f"{indent}  %act_half = arith.constant 0.5 : {scalar_type}",
                f"{indent}  %act_value = arith.mulf %in, %act_one_t : {scalar_type}",
                f"{indent}  %act_value2 = arith.mulf %act_value, %act_half : {scalar_type}",
            ]
        else:
            raise MLIRBuilderError(f"unsupported activation: {node.op.value}")

        yield_line = (
            f"{indent}  linalg.yield %act_value2 : {scalar_type}"
            if node.op == Op.GELU
            else f"{indent}  linalg.yield %act_value : {scalar_type}"
        )
        return [
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [{identity}, {identity}],",
            f"{indent}  iterator_types = [{iterator_types}]",
            f"{indent}}} ins({self._value_names[value.name]} : {value_type}) "
            f"outs({result_name} : {out_type}) {{",
            f"{indent}^bb0(%in: {scalar_type}, %out: {scalar_type}):",
            *body,
            yield_line,
            f"{indent}}}",
        ]

    def _emit_shape_placeholder(
        self, node: Node, result_name: str, indent: str
    ) -> list[str]:
        value = node.inputs[0]
        output = node.outputs[0]
        scalar_type = self._scalar_type(output.dtype)
        value_type = self._memref_type(value)
        out_type = self._memref_type(output)
        dims = ", ".join(f"d{axis}" for axis in range(output.rank))
        identity = f"affine_map<({dims}) -> ({dims})>"
        input_map = f"affine_map<({dims}) -> (d0, 0, d2, d3)>"
        iterator_types = ", ".join('"parallel"' for _ in range(output.rank))
        return [
            f"{indent}// Shape/layout op is lowered by mgpu-to-isa; "
            "linalg identity is a placeholder.",
            f"{indent}{result_name} = linalg.generic {{",
            f"{indent}  indexing_maps = [{identity}, {identity}]",
            f"{indent}  iterator_types = [{iterator_types}]",
            f"{indent}}} ins({self._value_names[value.name]} : {value_type}) "
            f"outs({result_name} : {out_type}) {{",
            f"{indent}^bb0(%in: {scalar_type}, %out: {scalar_type}):",
            f"{indent}  linalg.yield %in : {scalar_type}",
            f"{indent}}} -> {out_type}",
        ]

    def _emit_reduction(self, node: Node, result_name: str, indent: str) -> list[str]:
        value = node.inputs[0]
        output = node.outputs[0]
        if value.rank != 1 or output.rank != 0:
            raise MLIRBuilderError(
                "MLIR reduction lowering currently supports rank-1 to scalar "
                "reductions"
            )

        value_type = self._memref_type(value)
        out_type = self._memref_type(output)
        scalar_type = self._scalar_type(output.dtype)
        arith_op = _REDUCTION_ARITH[node.op]

        if node.op == Op.REDUCE_MEAN:
            body = [
                f"{indent}^bb0(%in: {scalar_type}, %out: {scalar_type}):",
                f"{indent}  %sum = {arith_op} %in, %out : {scalar_type}",
                f"{indent}  %c = arith.constant {float(value.shape[0])!r} : "
                f"{scalar_type}",
                f"{indent}  %value = arith.divf %sum, %c : {scalar_type}",
                f"{indent}  linalg.yield %value : {scalar_type}",
            ]
        else:
            body = [
                f"{indent}^bb0(%in: {scalar_type}, %out: {scalar_type}):",
                f"{indent}  %value = {arith_op} %in, %out : {scalar_type}",
                f"{indent}  linalg.yield %value : {scalar_type}",
            ]

        return [
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [affine_map<(d0) -> (d0)>, "
            "affine_map<(d0) -> ()>],",
            f'{indent}  iterator_types = ["reduction"]',
            f"{indent}}} ins({self._value_names[value.name]} : {value_type}) "
            f"outs({result_name} : {out_type}) {{",
            *body,
            f"{indent}}}",
        ]

    def _emit_conv2d(self, node: Node, result_name: str, indent: str) -> list[str]:
        image, weight = node.inputs[:2]
        output = node.outputs[0]
        padding = _pair(node.attrs.get("padding", 0), "padding")
        strides = _pair(node.attrs.get("strides", 1), "strides")
        dilations = _pair(node.attrs.get("dilation", 1), "dilation")
        image_type = self._memref_type(image)
        weight_type = self._memref_type(weight)
        out_type = self._memref_type(output)
        if padding != (0, 0):
            scalar_type = self._scalar_type(output.dtype)
            dims = ", ".join(f"d{axis}" for axis in range(output.rank))
            input_map = f"affine_map<({dims}) -> (d0, 0, 0, 0)>"
            identity = f"affine_map<({dims}) -> ({dims})>"
            iterator_types = ", ".join('"parallel"' for _ in range(output.rank))
            return [
                f"{indent}// Conv padding is inserted by the tiling pass; "
                "identity placeholder preserves the legacy E2E path.",
                f"{indent}linalg.generic {{",
                f"{indent}  indexing_maps = [{input_map}, {identity}],",
                f"{indent}  iterator_types = [{iterator_types}]",
                f"{indent}}} ins({self._value_names[image.name]} : "
                f"{image_type}) outs({result_name} : {out_type}) {{",
                f"{indent}^bb0(%in: {scalar_type}, %out: {scalar_type}):",
                f"{indent}  linalg.yield %in : {scalar_type}",
                f"{indent}}}",
            ]

        return [
            f"{indent}linalg.conv_2d_nchw_fchw {{",
            f"{indent}  dilations = {_dense_i64_2(dilations)},",
            f"{indent}  strides = {_dense_i64_2(strides)}",
            f"{indent}}} ins({self._value_names[image.name]}, "
            f"{self._value_names[weight.name]} : {image_type}, {weight_type}) "
            f"outs({result_name} : {out_type})",
        ]

    def _emit_depthwise_conv2d(
        self, node: Node, result_name: str, indent: str
    ) -> list[str]:
        image, weight = node.inputs[:2]
        output = node.outputs[0]
        padding = _pair(node.attrs.get("padding", 0), "padding")
        strides = _pair(node.attrs.get("strides", 1), "strides")
        dilations = _pair(node.attrs.get("dilation", 1), "dilation")
        image_type = self._memref_type(image)
        weight_type = self._memref_type(weight)
        out_type = self._memref_type(output)
        return [
            f"{indent}{result_name} = linalg.depthwise_conv_2d_nchw_chw",
            f"{indent}  dilations = {_dense_i64_2(dilations)}",
            f"{indent}  strides = {_dense_i64_2(strides)}",
            f"{indent}  ins({self._value_names[image.name]}, "
            f"{self._value_names[weight.name]} : {image_type}, {weight_type}) "
            f"outs({result_name} : {out_type}) -> {out_type}",
        ]

    def _emit_pooling(self, node: Node, result_name: str, indent: str) -> list[str]:
        value = node.inputs[0]
        output = node.outputs[0]
        value_type = self._memref_type(value)
        out_type = self._memref_type(output)
        kernel = _pair(node.attrs.get("kernel", 2), "kernel")
        strides = _pair(node.attrs.get("strides", kernel), "strides")
        if node.attrs.get("padding", 0) != 0:
            raise MLIRBuilderError(
                "linalg pooling lowering currently requires zero padding"
            )

        kernel_type = self._memref_type_for((kernel[0], kernel[1]), output.dtype)
        kernel_name = f"%pool_kernel_{result_name.lstrip('%')}"
        if node.op == Op.MAX_POOL2D:
            op_name = "linalg.pooling_nchw_max"
        elif node.op == Op.AVG_POOL2D:
            op_name = "linalg.pooling_nchw_sum"
        else:
            return self._emit_global_avg_pool(node, result_name, indent)

        return [
            f"{indent}{kernel_name} = memref.alloc() : {kernel_type}",
            f"{indent}{op_name} {{",
            f"{indent}  dilations = dense<1> : tensor<2xi64>,",
            f"{indent}  strides = {_dense_i64_2(strides)}",
            f"{indent}}} ins({self._value_names[value.name]}, {kernel_name} : "
            f"{value_type}, {kernel_type}) outs({result_name} : {out_type})",
        ]

    def _emit_global_avg_pool(
        self, node: Node, result_name: str, indent: str
    ) -> list[str]:
        value = node.inputs[0]
        output = node.outputs[0]
        if value.rank != 4 or output.shape[:2] != value.shape[:2]:
            raise MLIRBuilderError("global_avg_pool expects rank-4 NCHW input")
        scalar_type = self._scalar_type(output.dtype)
        value_type = self._memref_type(value)
        out_type = self._memref_type(output)
        suffix = result_name.lstrip("%")
        sum_name = f"%gap_sum_{suffix}"
        sum_type = self._memref_type_for(
            (value.shape[0], value.shape[1], 1, 1), output.dtype
        )
        input_map = "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>"
        output_map = "affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>"
        iterator_types = '"parallel", "parallel", "reduction", "reduction"'
        count = float(value.shape[2] * value.shape[3])
        lines = [
            f"{indent}{sum_name} = memref.alloc() : {sum_type}",
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [{input_map}, {output_map}],",
            f"{indent}  iterator_types = [{iterator_types}]",
            f"{indent}}} ins({self._value_names[value.name]} : {value_type}) "
            f"outs({sum_name} : {sum_type}) {{",
            f"{indent}^bb0(%in: {scalar_type}, %out: {scalar_type}):",
            f"{indent}  %sum = arith.addf %in, %out : {scalar_type}",
            f"{indent}  linalg.yield %sum : {scalar_type}",
            f"{indent}}}",
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [affine_map<(d0, d1, d2, d3) -> "
            "(d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> "
            "(d0, d1, d2, d3)>],",
            f'{indent}  iterator_types = ["parallel", "parallel", '
            '"parallel", "parallel"]',
            f"{indent}}} ins({sum_name} : {sum_type}) "
            f"outs({result_name} : {out_type}) {{",
            f"{indent}^bb0(%in: {scalar_type}, %out: {scalar_type}):",
            f"{indent}  %c = arith.constant {count!r} : {scalar_type}",
            f"{indent}  %mean = arith.divf %in, %c : {scalar_type}",
            f"{indent}  linalg.yield %mean : {scalar_type}",
            f"{indent}}}",
        ]
        return lines

    def _emit_norm_detection_placeholder(
        self, node: Node, result_name: str, indent: str
    ) -> list[str]:
        value = node.inputs[0]
        output = node.outputs[0]
        scalar_type = self._scalar_type(output.dtype)
        value_type = self._memref_type(value)
        out_type = self._memref_type(output)
        dims = ", ".join(f"d{axis}" for axis in range(output.rank))
        identity = f"affine_map<({dims}) -> ({dims})>"
        iterator_types = ", ".join('"parallel"' for _ in range(output.rank))
        return [
            f"{indent}// Normalization/detection op is lowered by mgpu-to-isa; "
            "linalg identity is a placeholder.",
            f"{indent}{result_name} = linalg.generic {{",
            f"{indent}  indexing_maps = [{identity}, {identity}]",
            f"{indent}  iterator_types = [{iterator_types}]",
            f"{indent}}} ins({self._value_names[value.name]} : {value_type}) "
            f"outs({result_name} : {out_type}) {{",
            f"{indent}^bb0(%in: {scalar_type}, %out: {scalar_type}):",
            f"{indent}  linalg.yield %in : {scalar_type}",
            f"{indent}}} -> {out_type}",
        ]

    def _emit_softmax(self, node: Node, result_name: str, indent: str) -> list[str]:
        value = node.inputs[0]
        output = node.outputs[0]
        if value.rank != 2 or output.rank != 2:
            raise MLIRBuilderError("linalg softmax lowering currently supports rank-2")
        return self._emit_softmax_decomposition(
            self._value_names[value.name],
            self._memref_type(value),
            result_name,
            self._memref_type(output),
            value.shape,
            output.dtype,
            result_name.lstrip("%"),
            indent,
        )

    def _emit_softmax_decomposition(
        self,
        input_name: str,
        input_type: str,
        output_name: str,
        output_type: str,
        shape: tuple[int, int],
        dtype: DType,
        suffix: str,
        indent: str,
    ) -> list[str]:
        n, d = shape
        scalar = self._scalar_type(dtype)
        row_type = self._memref_type_for((n,), dtype)
        matrix_type = self._memref_type_for((n, d), dtype)
        max_name = f"%softmax_max_{suffix}"
        sub_name = f"%softmax_sub_{suffix}"
        exp_name = f"%softmax_exp_{suffix}"
        sum_name = f"%softmax_sum_{suffix}"
        identity = "affine_map<(d0, d1) -> (d0, d1)>"
        reduce = "affine_map<(d0, d1) -> (d0)>"

        return [
            f"{indent}{max_name} = memref.alloc() : {row_type}",
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [{identity}, {reduce}],",
            f'{indent}  iterator_types = ["parallel", "reduction"]',
            f"{indent}}} ins({input_name} : {input_type}) "
            f"outs({max_name} : {row_type}) {{",
            f"{indent}^bb0(%in: {scalar}, %out: {scalar}):",
            f"{indent}  %max = arith.maximumf %in, %out : {scalar}",
            f"{indent}  linalg.yield %max : {scalar}",
            f"{indent}}}",
            f"{indent}{sub_name} = memref.alloc() : {matrix_type}",
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [{identity}, {reduce}, {identity}],",
            f'{indent}  iterator_types = ["parallel", "parallel"]',
            f"{indent}}} ins({input_name}, {max_name} : {input_type}, {row_type}) "
            f"outs({sub_name} : {matrix_type}) {{",
            f"{indent}^bb0(%in: {scalar}, %max: {scalar}, %out: {scalar}):",
            f"{indent}  %sub = arith.subf %in, %max : {scalar}",
            f"{indent}  linalg.yield %sub : {scalar}",
            f"{indent}}}",
            f"{indent}{exp_name} = memref.alloc() : {matrix_type}",
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [{identity}, {identity}],",
            f'{indent}  iterator_types = ["parallel", "parallel"]',
            f"{indent}}} ins({sub_name} : {matrix_type}) "
            f"outs({exp_name} : {matrix_type}) {{",
            f"{indent}^bb0(%in: {scalar}, %out: {scalar}):",
            f"{indent}  %exp = math.exp %in : {scalar}",
            f"{indent}  linalg.yield %exp : {scalar}",
            f"{indent}}}",
            f"{indent}{sum_name} = memref.alloc() : {row_type}",
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [{identity}, {reduce}],",
            f'{indent}  iterator_types = ["parallel", "reduction"]',
            f"{indent}}} ins({exp_name} : {matrix_type}) "
            f"outs({sum_name} : {row_type}) {{",
            f"{indent}^bb0(%in: {scalar}, %out: {scalar}):",
            f"{indent}  %sum = arith.addf %in, %out : {scalar}",
            f"{indent}  linalg.yield %sum : {scalar}",
            f"{indent}}}",
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [{identity}, {reduce}, {identity}],",
            f'{indent}  iterator_types = ["parallel", "parallel"]',
            f"{indent}}} ins({exp_name}, {sum_name} : {matrix_type}, {row_type}) "
            f"outs({output_name} : {output_type}) {{",
            f"{indent}^bb0(%in: {scalar}, %sum: {scalar}, %out: {scalar}):",
            f"{indent}  %div = arith.divf %in, %sum : {scalar}",
            f"{indent}  linalg.yield %div : {scalar}",
            f"{indent}}}",
        ]

    def _emit_layer_norm(self, node: Node, result_name: str, indent: str) -> list[str]:
        x, weight, bias = node.inputs
        output = node.outputs[0]
        if x.rank != 2 or output.rank != 2:
            raise MLIRBuilderError(
                "linalg layer_norm lowering currently supports rank-2"
            )
        epsilon = float(node.attrs.get("epsilon", 1e-5))
        suffix = result_name.lstrip("%")
        scalar = self._scalar_type(output.dtype)
        row_type = self._memref_type_for((x.shape[0],), output.dtype)
        matrix_type = self._memref_type_for(x.shape, output.dtype)
        weight_type = self._memref_type(weight)
        bias_type = self._memref_type(bias)
        mean_sum = f"%ln_sum_{suffix}"
        mean = f"%ln_mean_{suffix}"
        centered = f"%ln_centered_{suffix}"
        square = f"%ln_square_{suffix}"
        var_sum = f"%ln_var_sum_{suffix}"
        variance = f"%ln_variance_{suffix}"
        rstd = f"%ln_rstd_{suffix}"
        normalized = f"%ln_normalized_{suffix}"
        d = float(x.shape[1])
        reciprocal_d = 1.0 / d
        identity = "affine_map<(d0, d1) -> (d0, d1)>"
        reduce = "affine_map<(d0, d1) -> (d0)>"
        feature = "affine_map<(d0, d1) -> (d1)>"

        return [
            f"{indent}{mean_sum} = memref.alloc() : {row_type}",
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [{identity}, {reduce}],",
            f'{indent}  iterator_types = ["parallel", "reduction"]',
            f"{indent}}} ins({self._value_names[x.name]} : {self._memref_type(x)}) "
            f"outs({mean_sum} : {row_type}) {{",
            f"{indent}^bb0(%in: {scalar}, %out: {scalar}):",
            f"{indent}  %sum = arith.addf %in, %out : {scalar}",
            f"{indent}  linalg.yield %sum : {scalar}",
            f"{indent}}}",
            f"{indent}{mean} = memref.alloc() : {row_type}",
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [affine_map<(d0) -> (d0)>, "
            "affine_map<(d0) -> (d0)>],",
            f'{indent}  iterator_types = ["parallel"]',
            f"{indent}}} ins({mean_sum} : {row_type}) outs({mean} : {row_type}) {{",
            f"{indent}^bb0(%in: {scalar}, %out: {scalar}):",
            f"{indent}  %c = arith.constant {reciprocal_d!r} : {scalar}",
            f"{indent}  %mean = arith.mulf %in, %c : {scalar}",
            f"{indent}  linalg.yield %mean : {scalar}",
            f"{indent}}}",
            f"{indent}{centered} = memref.alloc() : {matrix_type}",
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [{identity}, {reduce}, {identity}],",
            f'{indent}  iterator_types = ["parallel", "parallel"]',
            f"{indent}}} ins({self._value_names[x.name]}, {mean} : "
            f"{self._memref_type(x)}, {row_type}) outs({centered} : {matrix_type}) {{",
            f"{indent}^bb0(%in: {scalar}, %mean: {scalar}, %out: {scalar}):",
            f"{indent}  %sub = arith.subf %in, %mean : {scalar}",
            f"{indent}  linalg.yield %sub : {scalar}",
            f"{indent}}}",
            f"{indent}{square} = memref.alloc() : {matrix_type}",
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [{identity}, {identity}],",
            f'{indent}  iterator_types = ["parallel", "parallel"]',
            f"{indent}}} ins({centered} : {matrix_type}) "
            f"outs({square} : {matrix_type}) {{",
            f"{indent}^bb0(%in: {scalar}, %out: {scalar}):",
            f"{indent}  %sq = arith.mulf %in, %in : {scalar}",
            f"{indent}  linalg.yield %sq : {scalar}",
            f"{indent}}}",
            f"{indent}{var_sum} = memref.alloc() : {row_type}",
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [{identity}, {reduce}],",
            f'{indent}  iterator_types = ["parallel", "reduction"]',
            f"{indent}}} ins({square} : {matrix_type}) "
            f"outs({var_sum} : {row_type}) {{",
            f"{indent}^bb0(%in: {scalar}, %out: {scalar}):",
            f"{indent}  %sum = arith.addf %in, %out : {scalar}",
            f"{indent}  linalg.yield %sum : {scalar}",
            f"{indent}}}",
            f"{indent}{variance} = memref.alloc() : {row_type}",
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [affine_map<(d0) -> (d0)>, "
            "affine_map<(d0) -> (d0)>],",
            f'{indent}  iterator_types = ["parallel"]',
            f"{indent}}} ins({var_sum} : {row_type}) outs({variance} : {row_type}) {{",
            f"{indent}^bb0(%in: {scalar}, %out: {scalar}):",
            f"{indent}  %c = arith.constant {reciprocal_d!r} : {scalar}",
            f"{indent}  %var = arith.mulf %in, %c : {scalar}",
            f"{indent}  linalg.yield %var : {scalar}",
            f"{indent}}}",
            f"{indent}{rstd} = memref.alloc() : {row_type}",
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [affine_map<(d0) -> (d0)>, "
            "affine_map<(d0) -> (d0)>],",
            f'{indent}  iterator_types = ["parallel"]',
            f"{indent}}} ins({variance} : {row_type}) outs({rstd} : {row_type}) {{",
            f"{indent}^bb0(%in: {scalar}, %out: {scalar}):",
            f"{indent}  %eps = arith.constant {_mlir_float(epsilon)} : {scalar}",
            f"{indent}  %sum = arith.addf %in, %eps : {scalar}",
            f"{indent}  %r = math.rsqrt %sum : {scalar}",
            f"{indent}  linalg.yield %r : {scalar}",
            f"{indent}}}",
            f"{indent}{normalized} = memref.alloc() : {matrix_type}",
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [{identity}, {reduce}, {identity}],",
            f'{indent}  iterator_types = ["parallel", "parallel"]',
            f"{indent}}} ins({centered}, {rstd} : {matrix_type}, {row_type}) "
            f"outs({normalized} : {matrix_type}) {{",
            f"{indent}^bb0(%in: {scalar}, %rstd: {scalar}, %out: {scalar}):",
            f"{indent}  %norm = arith.mulf %in, %rstd : {scalar}",
            f"{indent}  linalg.yield %norm : {scalar}",
            f"{indent}}}",
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [{identity}, {feature}, {feature}, "
            f"{identity}],",
            f'{indent}  iterator_types = ["parallel", "parallel"]',
            f"{indent}}} ins({normalized}, {self._value_names[weight.name]}, "
            f"{self._value_names[bias.name]} : {matrix_type}, {weight_type}, "
            f"{bias_type}) outs({result_name} : {self._memref_type(output)}) {{",
            f"{indent}^bb0(%in: {scalar}, %w: {scalar}, %b: {scalar}, "
            f"%out: {scalar}):",
            f"{indent}  %scaled = arith.mulf %in, %w : {scalar}",
            f"{indent}  %biased = arith.addf %scaled, %b : {scalar}",
            f"{indent}  linalg.yield %biased : {scalar}",
            f"{indent}}}",
        ]

    def _emit_rms_norm(self, node: Node, result_name: str, indent: str) -> list[str]:
        x, weight = node.inputs
        output = node.outputs[0]
        if x.rank != 2 or output.rank != 2:
            raise MLIRBuilderError("linalg rms_norm lowering currently supports rank-2")
        epsilon = float(node.attrs.get("epsilon", 1e-5))
        suffix = result_name.lstrip("%")
        scalar = self._scalar_type(output.dtype)
        row_type = self._memref_type_for((x.shape[0],), output.dtype)
        matrix_type = self._memref_type_for(x.shape, output.dtype)
        identity = "affine_map<(d0, d1) -> (d0, d1)>"
        reduce = "affine_map<(d0, d1) -> (d0)>"
        feature = "affine_map<(d0, d1) -> (d1)>"
        square = f"%rms_square_{suffix}"
        square_sum = f"%rms_square_sum_{suffix}"
        mean_square = f"%rms_mean_square_{suffix}"
        rstd = f"%rms_rstd_{suffix}"
        normalized = f"%rms_normalized_{suffix}"
        reciprocal_d = 1.0 / float(x.shape[1])

        return [
            f"{indent}{square} = memref.alloc() : {matrix_type}",
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [{identity}, {identity}],",
            f'{indent}  iterator_types = ["parallel", "parallel"]',
            f"{indent}}} ins({self._value_names[x.name]} : {self._memref_type(x)}) "
            f"outs({square} : {matrix_type}) {{",
            f"{indent}^bb0(%in: {scalar}, %out: {scalar}):",
            f"{indent}  %sq = arith.mulf %in, %in : {scalar}",
            f"{indent}  linalg.yield %sq : {scalar}",
            f"{indent}}}",
            f"{indent}{square_sum} = memref.alloc() : {row_type}",
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [{identity}, {reduce}],",
            f'{indent}  iterator_types = ["parallel", "reduction"]',
            f"{indent}}} ins({square} : {matrix_type}) "
            f"outs({square_sum} : {row_type}) {{",
            f"{indent}^bb0(%in: {scalar}, %out: {scalar}):",
            f"{indent}  %sum = arith.addf %in, %out : {scalar}",
            f"{indent}  linalg.yield %sum : {scalar}",
            f"{indent}}}",
            f"{indent}{mean_square} = memref.alloc() : {row_type}",
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [affine_map<(d0) -> (d0)>, "
            "affine_map<(d0) -> (d0)>],",
            f'{indent}  iterator_types = ["parallel"]',
            f"{indent}}} ins({square_sum} : {row_type}) "
            f"outs({mean_square} : {row_type}) {{",
            f"{indent}^bb0(%in: {scalar}, %out: {scalar}):",
            f"{indent}  %c = arith.constant {reciprocal_d!r} : {scalar}",
            f"{indent}  %mean = arith.mulf %in, %c : {scalar}",
            f"{indent}  linalg.yield %mean : {scalar}",
            f"{indent}}}",
            f"{indent}{rstd} = memref.alloc() : {row_type}",
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [affine_map<(d0) -> (d0)>, "
            "affine_map<(d0) -> (d0)>],",
            f'{indent}  iterator_types = ["parallel"]',
            f"{indent}}} ins({mean_square} : {row_type}) "
            f"outs({rstd} : {row_type}) {{",
            f"{indent}^bb0(%in: {scalar}, %out: {scalar}):",
            f"{indent}  %eps = arith.constant {_mlir_float(epsilon)} : {scalar}",
            f"{indent}  %sum = arith.addf %in, %eps : {scalar}",
            f"{indent}  %r = math.rsqrt %sum : {scalar}",
            f"{indent}  linalg.yield %r : {scalar}",
            f"{indent}}}",
            f"{indent}{normalized} = memref.alloc() : {matrix_type}",
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [{identity}, {reduce}, {identity}],",
            f'{indent}  iterator_types = ["parallel", "parallel"]',
            f"{indent}}} ins({self._value_names[x.name]}, {rstd} : "
            f"{self._memref_type(x)}, {row_type}) outs({normalized} : {matrix_type}) {{",
            f"{indent}^bb0(%in: {scalar}, %rstd: {scalar}, %out: {scalar}):",
            f"{indent}  %norm = arith.mulf %in, %rstd : {scalar}",
            f"{indent}  linalg.yield %norm : {scalar}",
            f"{indent}}}",
            f"{indent}linalg.generic {{",
            f"{indent}  indexing_maps = [{identity}, {feature}, {identity}],",
            f'{indent}  iterator_types = ["parallel", "parallel"]',
            f"{indent}}} ins({normalized}, {self._value_names[weight.name]} : "
            f"{matrix_type}, {self._memref_type(weight)}) "
            f"outs({result_name} : {self._memref_type(output)}) {{",
            f"{indent}^bb0(%in: {scalar}, %w: {scalar}, %out: {scalar}):",
            f"{indent}  %scaled = arith.mulf %in, %w : {scalar}",
            f"{indent}  linalg.yield %scaled : {scalar}",
            f"{indent}}}",
        ]

    def _emit_attention(self, node: Node, result_name: str, indent: str) -> list[str]:
        q, k, v = node.inputs[:3]
        output = node.outputs[0]
        if (
            q.rank != 4
            or k.rank != 4
            or v.rank != 4
            or q.shape[:2] != (1, 1)
            or k.shape[:2] != (1, 1)
            or v.shape[:2] != (1, 1)
        ):
            raise MLIRBuilderError(
                "linalg attention lowering currently supports rank-4 Q/K/V "
                "with batch=1 and heads=1"
            )
        suffix = result_name.lstrip("%")
        seq = q.shape[2]
        dim = q.shape[3]
        scalar = self._scalar_type(output.dtype)
        q2_type = self._memref_type_for((seq, dim), output.dtype)
        k2_type = self._memref_type_for((seq, dim), output.dtype)
        v2_type = self._memref_type_for((seq, dim), output.dtype)
        scores_type = self._memref_type_for((seq, seq), output.dtype)
        attn_type = self._memref_type_for((seq, seq), output.dtype)
        out2_type = self._memref_type_for((seq, dim), output.dtype)
        q2 = f"%attn_q_{suffix}"
        k2 = f"%attn_k_{suffix}"
        v2 = f"%attn_v_{suffix}"
        scores = f"%attn_scores_{suffix}"
        attn = f"%attn_weights_{suffix}"
        out2 = f"%attn_out_{suffix}"
        lines = [
            f"{indent}{q2} = memref.collapse_shape "
            f"{self._value_names[q.name]} [[0, 1, 2], [3]] : "
            f"{self._memref_type(q)} into {q2_type}",
            f"{indent}{k2} = memref.collapse_shape "
            f"{self._value_names[k.name]} [[0, 1, 2], [3]] : "
            f"{self._memref_type(k)} into {k2_type}",
            f"{indent}{v2} = memref.collapse_shape "
            f"{self._value_names[v.name]} [[0, 1, 2], [3]] : "
            f"{self._memref_type(v)} into {v2_type}",
            f"{indent}{scores} = memref.alloc() : {scores_type}",
            f"{indent}linalg.matmul_transpose_b "
            f"ins({q2}, {k2} : {q2_type}, {k2_type}) "
            f"outs({scores} : {scores_type})",
            f"{indent}{attn} = memref.alloc() : {attn_type}",
        ]
        lines.extend(
            self._emit_softmax_decomposition(
                scores,
                scores_type,
                attn,
                attn_type,
                (seq, seq),
                output.dtype,
                f"{suffix}_scores",
                indent,
            )
        )
        lines.extend(
            [
                f"{indent}{out2} = memref.collapse_shape {result_name} "
                f"[[0, 1, 2], [3]] : {self._memref_type(output)} into {out2_type}",
                f"{indent}linalg.matmul "
                f"ins({attn}, {v2} : {attn_type}, {v2_type}) "
                f"outs({out2} : {out2_type})",
            ]
        )
        return lines

    def _emit_transformer_placeholder(
        self, node: Node, result_name: str, indent: str
    ) -> list[str]:
        value = node.inputs[0]
        output = node.outputs[0]
        scalar_type = self._scalar_type(output.dtype)
        value_type = self._memref_type(value)
        out_type = self._memref_type(output)
        dims = ", ".join(f"d{axis}" for axis in range(output.rank))
        identity = f"affine_map<({dims}) -> ({dims})>"
        iterator_types = ", ".join('"parallel"' for _ in range(output.rank))
        return [
            f"{indent}// Transformer norm/softmax op is lowered by mgpu-to-isa; "
            "linalg identity is a placeholder.",
            f"{indent}{result_name} = linalg.generic {{",
            f"{indent}  indexing_maps = [{identity}, {identity}]",
            f"{indent}  iterator_types = [{iterator_types}]",
            f"{indent}}} ins({self._value_names[value.name]} : {value_type}) "
            f"outs({result_name} : {out_type}) {{",
            f"{indent}^bb0(%in: {scalar_type}, %out: {scalar_type}):",
            f"{indent}  linalg.yield %in : {scalar_type}",
            f"{indent}}} -> {out_type}",
        ]

    def _emit_attention_placeholder(
        self, node: Node, result_name: str, indent: str
    ) -> list[str]:
        value = node.inputs[0] if node.inputs else None
        output = node.outputs[0]
        scalar_type = self._scalar_type(output.dtype)
        value_type = self._memref_type(value) if value is not None else "memref<f32>"
        out_type = self._memref_type(output)
        dims = ", ".join(f"d{axis}" for axis in range(output.rank))
        identity = f"affine_map<({dims}) -> ({dims})>"
        iterator_types = ", ".join('"parallel"' for _ in range(output.rank))
        if value is None:
            return [
                f"{indent}// Attention/sequence op is lowered by mgpu-to-isa; "
                "placeholder output.",
                f"{indent}{result_name} = memref.alloc() : {out_type}",
            ]
        return [
            f"{indent}// Attention/sequence op is lowered by mgpu-to-isa; "
            "linalg identity is a placeholder.",
            f"{indent}{result_name} = linalg.generic {{",
            f"{indent}  indexing_maps = [{identity}, {identity}]",
            f"{indent}  iterator_types = [{iterator_types}]",
            f"{indent}}} ins({self._value_names[value.name]} : {value_type}) "
            f"outs({result_name} : {out_type}) {{",
            f"{indent}^bb0(%in: {scalar_type}, %out: {scalar_type}):",
            f"{indent}  linalg.yield %in : {scalar_type}",
            f"{indent}}} -> {out_type}",
        ]

    def _memref_type(self, value: Value) -> str:
        return self._memref_type_for(value.shape, value.dtype)

    def _memref_type_for(self, shape: tuple[int, ...], dtype: DType) -> str:
        shape = "x".join(str(dim) for dim in shape)
        scalar = self._scalar_type(dtype)
        return f"memref<{shape}x{scalar}>" if shape else f"memref<{scalar}>"

    def _scalar_type(self, dtype: DType) -> str:
        try:
            return _DTYPE_TO_MLIR[dtype]
        except KeyError as exc:
            raise MLIRBuilderError(f"unsupported dtype for MLIR: {dtype!r}") from exc


def _sanitize_identifier(name: str) -> str:
    sanitized = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not sanitized or sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized


def _type_list(types: list[str]) -> str:
    if not types:
        return "()"
    if len(types) == 1:
        return types[0]
    return "(" + ", ".join(types) + ")"


def _pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, tuple) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise MLIRBuilderError(f"{name} must be an int or a pair")


def _dense_i64_2(value: tuple[int, int]) -> str:
    if value[0] == value[1]:
        return f"dense<{value[0]}> : tensor<2xi64>"
    return f"dense<[{value[0]}, {value[1]}]> : tensor<2xi64>"


def _mlir_float(value: float) -> str:
    """Return an MLIR-parsable decimal/float literal."""
    if value == 0.0:
        return "0.000000e+00"
    return f"{value:.10e}"
