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
            f"{indent}%tx_{suffix} = mgpu.thread_id x",
            f"{indent}%ty_{suffix} = mgpu.thread_id y",
            f"{indent}%tz_{suffix} = mgpu.thread_id z",
            f"{indent}%bx_{suffix} = mgpu.block_id x",
            f"{indent}%by_{suffix} = mgpu.block_id y",
            f"{indent}%bz_{suffix} = mgpu.block_id z",
            f"{indent}%gx_{suffix} = mgpu.grid_dim x",
            f"{indent}%gy_{suffix} = mgpu.grid_dim y",
            f"{indent}%gz_{suffix} = mgpu.grid_dim z",
            f"{indent}%bdx_{suffix} = mgpu.block_dim x",
            f"{indent}%bdy_{suffix} = mgpu.block_dim y",
            f"{indent}%bdz_{suffix} = mgpu.block_dim z",
        ]

    def _emit_compute(self, node: Node, result_name: str, indent: str) -> list[str]:
        if node.op in _BINARY_ARITH:
            return self._emit_binary_generic(node, result_name, indent)
        if node.op in _REDUCTION_ARITH:
            return self._emit_reduction(node, result_name, indent)
        if node.op == Op.MATMUL:
            return self._emit_matmul(node, result_name, indent)
        if node.op == Op.CONV2D:
            return self._emit_conv2d(node, result_name, indent)
        if node.op == Op.DEPTHWISE_CONV2D:
            return self._emit_depthwise_conv2d(node, result_name, indent)
        if node.op in {
            Op.MAX_POOL2D,
            Op.AVG_POOL2D,
            Op.GLOBAL_AVG_POOL,
        }:
            return self._emit_pooling_placeholder(node, result_name, indent)
        if node.op in {
            Op.BATCH_NORM,
            Op.GROUP_NORM,
            Op.BOX_DECODE,
            Op.NMS,
        }:
            return self._emit_norm_detection_placeholder(node, result_name, indent)
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
            f"{indent}{result_name} = linalg.generic {{",
            f"{indent}  indexing_maps = [{maps}]",
            f"{indent}  iterator_types = [{iterator_types}]",
            f"{indent}}} ins({self._value_names[left.name]}, "
            f"{self._value_names[right.name]} : {left_type}, {right_type}) "
            f"outs({result_name} : {out_type}) {{",
            f"{indent}^bb0(%lhs: {scalar_type}, %rhs: {scalar_type}, "
            f"%out: {scalar_type}):",
            f"{indent}  %value = {arith_op} %lhs, %rhs : {scalar_type}",
            f"{indent}  linalg.yield %value : {scalar_type}",
            f"{indent}}} -> {out_type}",
        ]

    def _emit_matmul(self, node: Node, result_name: str, indent: str) -> list[str]:
        left, right = node.inputs
        output = node.outputs[0]
        left_type = self._memref_type(left)
        right_type = self._memref_type(right)
        out_type = self._memref_type(output)
        return [
            f"{indent}{result_name} = linalg.matmul "
            f"ins({self._value_names[left.name]}, {self._value_names[right.name]} : "
            f"{left_type}, {right_type}) outs({result_name} : {out_type}) "
            f"-> {out_type}"
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
        iterator_types = ", ".join('"parallel"' for _ in range(output.rank))

        if node.op == Op.RELU:
            body = [
                f"{indent}  %c0 = arith.constant 0.000000e+00 : {scalar_type}",
                f"{indent}  %value = arith.maximumf %in, %c0 : {scalar_type}",
            ]
        elif node.op == Op.LEAKY_RELU:
            alpha = float(node.attrs.get("alpha", 0.01))
            body = [
                f"{indent}  %c0 = arith.constant 0.000000e+00 : {scalar_type}",
                f"{indent}  %neg = arith.cmpf olt, %in, %c0 : {scalar_type}",
                f"{indent}  %scaled = arith.mulf %in, " f"{alpha!r} : {scalar_type}",
                f"{indent}  %value = arith.select %neg, %scaled, %in : {scalar_type}",
            ]
        elif node.op == Op.SIGMOID:
            body = [
                f"{indent}  %neg = arith.negf %in : {scalar_type}",
                f"{indent}  %exp = math.exp %neg : {scalar_type}",
                f"{indent}  %one = arith.constant 1.000000e+00 : {scalar_type}",
                f"{indent}  %den = arith.addf %one, %exp : {scalar_type}",
                f"{indent}  %value = arith.divf %one, %den : {scalar_type}",
            ]
        elif node.op == Op.SILU:
            body = [
                f"{indent}  %neg = arith.negf %in : {scalar_type}",
                f"{indent}  %exp = math.exp %neg : {scalar_type}",
                f"{indent}  %one = arith.constant 1.000000e+00 : {scalar_type}",
                f"{indent}  %den = arith.addf %one, %exp : {scalar_type}",
                f"{indent}  %sigmoid = arith.divf %one, %den : {scalar_type}",
                f"{indent}  %value = arith.mulf %in, %sigmoid : {scalar_type}",
            ]
        elif node.op == Op.GELU:
            body = [
                f"{indent}  %cst = arith.constant 0.7978845608028654 : {scalar_type}",
                f"{indent}  %x2 = arith.mulf %in, %in : {scalar_type}",
                f"{indent}  %x3 = arith.mulf %x2, %in : {scalar_type}",
                f"{indent}  %cst2 = arith.constant 0.044715 : {scalar_type}",
                f"{indent}  %poly = arith.mulf %cst2, %x3 : {scalar_type}",
                f"{indent}  %sum = arith.addf %in, %poly : {scalar_type}",
                f"{indent}  %arg = arith.mulf %cst, %sum : {scalar_type}",
                f"{indent}  %tanh = math.tanh %arg : {scalar_type}",
                f"{indent}  %one = arith.constant 1.000000e+00 : {scalar_type}",
                f"{indent}  %one_t = arith.addf %one, %tanh : {scalar_type}",
                f"{indent}  %half = arith.constant 0.5 : {scalar_type}",
                f"{indent}  %value = arith.mulf %in, %one_t : {scalar_type}",
                f"{indent}  %value2 = arith.mulf %value, %half : {scalar_type}",
            ]
        else:
            raise MLIRBuilderError(f"unsupported activation: {node.op.value}")

        yield_line = (
            f"{indent}  linalg.yield %value2 : {scalar_type}"
            if node.op == Op.GELU
            else f"{indent}  linalg.yield %value : {scalar_type}"
        )
        return [
            f"{indent}{result_name} = linalg.generic {{",
            f"{indent}  indexing_maps = [{identity}, {identity}]",
            f"{indent}  iterator_types = [{iterator_types}]",
            f"{indent}}} ins({self._value_names[value.name]} : {value_type}) "
            f"outs({result_name} : {out_type}) {{",
            f"{indent}^bb0(%in: {scalar_type}, %out: {scalar_type}):",
            *body,
            yield_line,
            f"{indent}}} -> {out_type}",
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

        return [
            f"{indent}{result_name} = linalg.generic {{",
            f"{indent}  indexing_maps = [affine_map<(d0) -> (d0)>, "
            "affine_map<(d0) -> ()>]",
            f'{indent}  iterator_types = ["reduction"]',
            f"{indent}}} ins({self._value_names[value.name]} : {value_type}) "
            f"outs({result_name} : {out_type}) {{",
            f"{indent}^bb0(%in: {scalar_type}, %out: {scalar_type}):",
            f"{indent}  %value = {arith_op} %in, %out : {scalar_type}",
            f"{indent}  linalg.yield %value : {scalar_type}",
            f"{indent}}} -> {out_type}",
        ]

    def _emit_conv2d(self, node: Node, result_name: str, indent: str) -> list[str]:
        image, weight = node.inputs[:2]
        output = node.outputs[0]
        padding = node.attrs.get("padding", 0)
        if padding != 0:
            raise MLIRBuilderError(
                "MLIR conv2d lowering currently supports padding=0 only"
            )

        strides = _pair(node.attrs.get("strides", 1), "strides")
        dilations = _pair(node.attrs.get("dilation", 1), "dilation")
        image_type = self._memref_type(image)
        weight_type = self._memref_type(weight)
        out_type = self._memref_type(output)

        return [
            f"{indent}{result_name} = linalg.conv_2d_nchw_fchw",
            f"{indent}  dilations = {_dense_i64_2(dilations)}",
            f"{indent}  strides = {_dense_i64_2(strides)}",
            f"{indent}  ins({self._value_names[image.name]}, "
            f"{self._value_names[weight.name]} : {image_type}, {weight_type}) "
            f"outs({result_name} : {out_type}) -> {out_type}",
        ]

    def _emit_depthwise_conv2d(
        self, node: Node, result_name: str, indent: str
    ) -> list[str]:
        image, weight = node.inputs[:2]
        output = node.outputs[0]
        padding = node.attrs.get("padding", 0)
        if padding != 0:
            raise MLIRBuilderError(
                "MLIR depthwise conv lowering currently supports padding=0 only"
            )
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

    def _emit_pooling_placeholder(
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
            f"{indent}// Pooling op is lowered by mgpu-to-isa; "
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

    def _memref_type(self, value: Value) -> str:
        shape = "x".join(str(dim) for dim in value.shape)
        scalar = self._scalar_type(value.dtype)
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
