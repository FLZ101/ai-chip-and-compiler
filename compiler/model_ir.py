"""Language-neutral model graph IR.

Step 005 defines the shared graph representation used by every frontend.
The graph is intentionally small and explicit:

* ``Value`` carries a name, data type, and static shape.
* ``Node`` is an SSA operation with input values, output values, and attributes.
* ``Graph`` owns inputs, nodes, and outputs, and can validate itself.

The operator set is designed once for both CNN/object-detection and
Transformer/LLM workloads.  No frontend-specific CNN or Transformer fork is
created.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class DType(str, Enum):
    """Data types consumed by the frontend and emitted later by MLIR."""

    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"
    I32 = "i32"
    I64 = "i64"
    I8 = "i8"
    U8 = "u8"
    I4 = "i4"
    U4 = "u4"
    F8_E4M3 = "f8_e4m3"
    F8_E5M2 = "f8_e5m2"
    BOOL = "bool"


class Op(str, Enum):
    """Unified CNN and Transformer/LLM operator set."""

    # Data source/sink.
    INPUT = "input"
    CONSTANT = "constant"

    # Elementwise and activation.
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    FMA = "fma"
    MAX = "max"
    MIN = "min"
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    SILU = "silu"
    GELU = "gelu"
    SIGMOID = "sigmoid"
    HARD_SIGMOID = "hard_sigmoid"

    # Reduction.
    REDUCE_SUM = "reduce_sum"
    REDUCE_MEAN = "reduce_mean"
    REDUCE_MAX = "reduce_max"

    # Linear algebra.
    MATMUL = "matmul"
    LINEAR = "linear"
    BATCH_MATMUL = "batch_matmul"

    # Convolution and pooling.
    CONV2D = "conv2d"
    DEPTHWISE_CONV2D = "depthwise_conv2d"
    MAX_POOL2D = "max_pool2d"
    AVG_POOL2D = "avg_pool2d"
    GLOBAL_AVG_POOL = "global_avg_pool"
    EMBEDDING_GATHER = "embedding_gather"
    ARGMAX = "argmax"

    # Normalization.
    BATCH_NORM = "batch_norm"
    GROUP_NORM = "group_norm"
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"

    # Softmax and attention.
    SOFTMAX = "softmax"
    MASKED_SOFTMAX = "masked_softmax"
    SCALED_DOT_PRODUCT_ATTENTION = "scaled_dot_product_attention"
    MASKED_ATTENTION = "masked_attention"

    # Position and sequence.
    ROPE = "rope"
    KV_CACHE_APPEND = "kv_cache_append"
    KV_CACHE_GATHER = "kv_cache_gather"
    CAUSAL_MASK = "causal_mask"

    # Shape and layout.
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    CONCAT = "concat"
    SLICE = "slice"
    PAD = "pad"
    RESIZE = "resize"
    EXPAND = "expand"
    SQUEEZE = "squeeze"
    UNSQUEEZE = "unsqueeze"

    # Object-detection helpers.
    BOX_DECODE = "box_decode"
    NMS = "nms"


class ModelIRError(Exception):
    """Raised when a model graph is malformed."""


@dataclass(frozen=True)
class SourceLocation:
    """Optional source/debug metadata attached to values and nodes."""

    op: str = ""
    shape: tuple[int, ...] = ()
    dtype: DType | str = DType.F32
    pass_name: str = "model-ir"
    location: str = ""

    def format(self) -> str:
        dtype = self.dtype.value if isinstance(self.dtype, DType) else self.dtype
        shape_text = "x".join(str(dim) for dim in self.shape)
        return (
            f"[op={self.op or '?'}, shape={shape_text or '?'}, "
            f"dtype={dtype or '?'}, pass={self.pass_name or '?'}, "
            f"loc={self.location or '?'}]"
        )


@dataclass(frozen=True)
class Value:
    """A typed SSA value."""

    name: str
    dtype: DType
    shape: tuple[int, ...]
    source: SourceLocation | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ModelIRError("Value name must not be empty")
        if not isinstance(self.dtype, DType):
            raise ModelIRError(f"value {self.name} has invalid dtype {self.dtype!r}")
        if not isinstance(self.shape, tuple) or not all(
            isinstance(dim, int) and dim >= 0 for dim in self.shape
        ):
            raise ModelIRError(f"value {self.name} has invalid shape {self.shape!r}")

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


@dataclass
class Node:
    """One SSA operation in a graph."""

    name: str
    op: Op
    inputs: list[Value] = field(default_factory=list)
    outputs: list[Value] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    source: SourceLocation | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ModelIRError("Node name must not be empty")
        if not isinstance(self.op, Op):
            raise ModelIRError(f"node {self.name} has invalid op {self.op!r}")
        if not all(isinstance(value, Value) for value in self.inputs):
            raise ModelIRError(f"node {self.name} has a non-Value input")
        if not all(isinstance(value, Value) for value in self.outputs):
            raise ModelIRError(f"node {self.name} has a non-Value output")
        if not isinstance(self.attrs, dict):
            raise ModelIRError(f"node {self.name} has non-dict attrs")


@dataclass
class Graph:
    """Container for a validated, printable model graph."""

    name: str = "graph"
    inputs: list[Value] = field(default_factory=list)
    outputs: list[Value] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    _value_counter: int = field(default=0, init=False, repr=False)

    def add_input(self, name: str, dtype: DType | str, shape: tuple[int, ...]) -> Value:
        value = Value(
            name,
            _as_dtype(dtype),
            tuple(shape),
            SourceLocation(op="input", shape=tuple(shape), dtype=_as_dtype(dtype)),
        )
        if value.name in {existing.name for existing in self.inputs}:
            raise ModelIRError(f"duplicate graph input name: {value.name}")
        self.inputs.append(value)
        return value

    def add_node(
        self,
        op: Op | str,
        inputs: Iterable[Value],
        attrs: dict[str, Any] | None = None,
        *,
        name: str | None = None,
        source: SourceLocation | None = None,
    ) -> list[Value]:
        """Infer outputs for ``op`` and append a new SSA node.

        This is the main builder API.  The manual frontend and later MLIR
        lowering both use it so every graph is constructed through the same
        validation and shape-inference path.
        """
        operation = _as_op(op)
        input_values = list(inputs)
        node_attrs = dict(attrs or {})
        node_name = name or self._fresh_name(operation)
        output_specs = infer_node_outputs(operation, input_values, node_attrs)

        output_values = [
            Value(
                name=node_name if len(output_specs) == 1 else f"{node_name}_{index}",
                dtype=dtype,
                shape=shape,
            )
            for index, (dtype, shape) in enumerate(output_specs)
        ]

        node = Node(
            name=node_name,
            op=operation,
            inputs=input_values,
            outputs=output_values,
            attrs=node_attrs,
            source=source,
        )
        self.nodes.append(node)
        return output_values

    def mark_outputs(self, values: Iterable[Value]) -> None:
        self.outputs = list(values)

    def validate(self) -> None:
        _validate_graph(self)

    def format(self) -> str:
        return format_graph(self)

    def _fresh_name(self, op: Op) -> str:
        base = op.value.replace(".", "_")
        while True:
            candidate = f"{base}_{self._value_counter}"
            self._value_counter += 1
            if candidate not in {
                value.name for value in self.inputs
            } and candidate not in {
                value.name for node in self.nodes for value in node.outputs
            }:
                return candidate


def _as_dtype(dtype: DType | str) -> DType:
    if isinstance(dtype, DType):
        return dtype
    try:
        return DType(dtype)
    except ValueError as exc:
        raise ModelIRError(f"unknown dtype: {dtype!r}") from exc


def _as_op(op: Op | str) -> Op:
    if isinstance(op, Op):
        return op
    try:
        return Op(op)
    except ValueError as exc:
        raise ModelIRError(f"unknown op: {op!r}") from exc


def _as_shape(shape: Iterable[int] | tuple[int, ...]) -> tuple[int, ...]:
    result = tuple(int(dim) for dim in shape)
    if any(dim < 0 for dim in result):
        raise ModelIRError(f"shape contains a negative dimension: {result!r}")
    return result


def _pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, tuple) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ModelIRError(f"{name} must be an int or a pair")


def _broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    if not shapes:
        return ()
    rank = max(len(shape) for shape in shapes)
    padded = [(1,) * (rank - len(shape)) + shape for shape in shapes]
    output: list[int] = []
    for dims in zip(*padded):
        non_one = {dim for dim in dims if dim != 1}
        if len(non_one) > 1:
            raise ModelIRError(f"cannot broadcast shapes: {shapes!r}")
        output.append(next(iter(non_one), 1))
    return tuple(output)


def _promote_dtype(*dtypes: DType) -> DType:
    if not dtypes:
        return DType.F32
    unique = set(dtypes)
    if len(unique) == 1:
        return dtypes[0]
    if DType.F32 in unique:
        return DType.F32
    if any(
        dtype in {DType.F16, DType.BF16, DType.F8_E4M3, DType.F8_E5M2}
        for dtype in unique
    ):
        return DType.F32
    if DType.I64 in unique:
        return DType.I64
    if DType.I32 in unique:
        return DType.I32
    if any(dtype in {DType.I8, DType.U8, DType.I4, DType.U4} for dtype in unique):
        return DType.I32
    return DType.BOOL


def _normalize_axes(axes: Iterable[int] | None, rank: int) -> tuple[int, ...]:
    if axes is None:
        return tuple(range(rank))
    normalized: list[int] = []
    for axis in axes:
        value = int(axis)
        if value < 0:
            value += rank
        if value < 0 or value >= rank:
            raise ModelIRError(f"axis {axis} out of range for rank {rank}")
        if value in normalized:
            raise ModelIRError(f"duplicate reduction axis: {axis}")
        normalized.append(value)
    return tuple(normalized)


def _shape_of_python_value(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        first_shape = _shape_of_python_value(value[0])
        for item in value[1:]:
            if _shape_of_python_value(item) != first_shape:
                raise ModelIRError("ragged nested constants are not supported")
        return (len(value),) + first_shape
    return ()


def _conv_output_dim(
    size: int,
    kernel: int,
    *,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    numerator = size + 2 * padding - dilation * (kernel - 1) - 1
    return numerator // stride + 1


def _pool_output_shape(
    shape: tuple[int, ...],
    *,
    kernel: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> tuple[int, ...]:
    if len(shape) != 4:
        raise ModelIRError("pooling expects a rank-4 NCHW tensor")
    n, c, h, w = shape
    out_h = _conv_output_dim(
        h, kernel[0], stride=stride[0], padding=padding[0], dilation=1
    )
    out_w = _conv_output_dim(
        w, kernel[1], stride=stride[1], padding=padding[1], dilation=1
    )
    return n, c, out_h, out_w


def infer_node_outputs(
    op: Op | str,
    inputs: list[Value],
    attrs: dict[str, Any],
) -> list[tuple[DType, tuple[int, ...]]]:
    """Infer dtype and shape for one node's outputs."""
    operation = _as_op(op)
    values = list(inputs)

    def one(
        dtype: DType, shape: tuple[int, ...]
    ) -> list[tuple[DType, tuple[int, ...]]]:
        return [(dtype, _as_shape(shape))]

    def unary() -> list[tuple[DType, tuple[int, ...]]]:
        _require_arity(operation, values, 1, 1)
        return one(values[0].dtype, values[0].shape)

    def elementwise() -> list[tuple[DType, tuple[int, ...]]]:
        _require_arity(operation, values, 2, 2)
        dtype = _promote_dtype(values[0].dtype, values[1].dtype)
        shape = _broadcast_shapes(values[0].shape, values[1].shape)
        return one(dtype, shape)

    def reduction() -> list[tuple[DType, tuple[int, ...]]]:
        _require_arity(operation, values, 1, 1)
        rank = values[0].rank
        axes = _normalize_axes(attrs.get("axes"), rank)
        keepdims = bool(attrs.get("keepdims", False))
        shape = tuple(
            1 if axis in axes and keepdims else dim
            for axis, dim in enumerate(values[0].shape)
            if axis not in axes or keepdims
        )
        return one(values[0].dtype, shape)

    if operation in {Op.INPUT}:
        _require_arity(operation, values, 0, 0)
        return one(
            _as_dtype(attrs.get("dtype", DType.F32)),
            _as_shape(attrs.get("shape", ())),
        )

    if operation == Op.CONSTANT:
        _require_arity(operation, values, 0, 0)
        if "value" not in attrs:
            raise ModelIRError("constant requires a value attribute")
        dtype = _as_dtype(attrs.get("dtype", DType.F32))
        shape = _shape_of_python_value(attrs["value"])
        return one(dtype, shape)

    if operation in {
        Op.RELU,
        Op.LEAKY_RELU,
        Op.SILU,
        Op.GELU,
        Op.SIGMOID,
        Op.HARD_SIGMOID,
        Op.SOFTMAX,
    }:
        return unary()

    if operation == Op.MASKED_SOFTMAX:
        _require_arity(operation, values, 2, 2)
        return one(values[0].dtype, values[0].shape)

    if operation == Op.ROPE:
        _require_arity(operation, values, 3, 3)
        return one(values[0].dtype, values[0].shape)

    if operation in {Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MAX, Op.MIN}:
        return elementwise()

    if operation == Op.FMA:
        _require_arity(operation, values, 3, 3)
        dtype = _promote_dtype(values[0].dtype, values[1].dtype, values[2].dtype)
        shape = _broadcast_shapes(values[0].shape, values[1].shape, values[2].shape)
        return one(dtype, shape)

    if operation in {Op.REDUCE_SUM, Op.REDUCE_MEAN, Op.REDUCE_MAX}:
        return reduction()

    if operation in {Op.MATMUL, Op.BATCH_MATMUL}:
        _require_arity(operation, values, 2, 2)
        if values[0].rank < 2 or values[1].rank < 2:
            raise ModelIRError(f"{operation.value} expects rank >= 2 inputs")
        if values[0].shape[-1] != values[1].shape[-2]:
            raise ModelIRError(
                f"{operation.value} inner dimensions do not match: "
                f"{values[0].shape[-1]} != {values[1].shape[-2]}"
            )
        batch = _broadcast_shapes(values[0].shape[:-2], values[1].shape[:-2])
        shape = batch + (values[0].shape[-2], values[1].shape[-1])
        return one(_promote_dtype(values[0].dtype, values[1].dtype), shape)

    if operation == Op.LINEAR:
        _require_arity(operation, values, 2, 3)
        x, weight = values[0], values[1]
        if x.rank < 2 or weight.rank != 2:
            raise ModelIRError("linear expects x with rank >= 2 and weight rank 2")
        if x.shape[-1] != weight.shape[1]:
            raise ModelIRError("linear input and weight shapes do not match")
        if len(values) == 3 and values[2].shape != (weight.shape[0],):
            raise ModelIRError("linear bias must have shape (out_features,)")
        dtype = _promote_dtype(
            x.dtype, weight.dtype, *([values[2].dtype] if len(values) == 3 else [])
        )
        return one(dtype, x.shape[:-1] + (weight.shape[0],))

    if operation == Op.CONV2D:
        _require_arity(operation, values, 2, 3)
        x, weight = values[0], values[1]
        if x.rank != 4 or weight.rank != 4:
            raise ModelIRError("conv2d expects rank-4 NCHW input and OIHW weight")
        groups = int(attrs.get("groups", 1))
        stride = _pair(attrs.get("strides", 1), "strides")
        padding = _pair(attrs.get("padding", 0), "padding")
        dilation = _pair(attrs.get("dilation", 1), "dilation")
        if x.shape[1] != weight.shape[1] * groups:
            raise ModelIRError("conv2d input channels do not match grouped weight")
        if weight.shape[0] % groups != 0:
            raise ModelIRError("conv2d output channels must be divisible by groups")
        out_h = _conv_output_dim(
            x.shape[2],
            weight.shape[2],
            stride=stride[0],
            padding=padding[0],
            dilation=dilation[0],
        )
        out_w = _conv_output_dim(
            x.shape[3],
            weight.shape[3],
            stride=stride[1],
            padding=padding[1],
            dilation=dilation[1],
        )
        return one(
            _promote_dtype(x.dtype, weight.dtype),
            (x.shape[0], weight.shape[0], out_h, out_w),
        )

    if operation == Op.DEPTHWISE_CONV2D:
        _require_arity(operation, values, 2, 3)
        x, weight = values[0], values[1]
        if x.rank != 4 or weight.rank != 4 or weight.shape[1] != 1:
            raise ModelIRError(
                "depthwise_conv2d expects NCHW input and (C,1,KH,KW) weight"
            )
        stride = _pair(attrs.get("strides", 1), "strides")
        padding = _pair(attrs.get("padding", 0), "padding")
        dilation = _pair(attrs.get("dilation", 1), "dilation")
        out_h = _conv_output_dim(
            x.shape[2],
            weight.shape[2],
            stride=stride[0],
            padding=padding[0],
            dilation=dilation[0],
        )
        out_w = _conv_output_dim(
            x.shape[3],
            weight.shape[3],
            stride=stride[1],
            padding=padding[1],
            dilation=dilation[1],
        )
        return one(
            _promote_dtype(x.dtype, weight.dtype),
            (x.shape[0], x.shape[1], out_h, out_w),
        )

    if operation in {Op.MAX_POOL2D, Op.AVG_POOL2D}:
        _require_arity(operation, values, 1, 1)
        kernel = _pair(attrs.get("kernel", 2), "kernel")
        stride = _pair(attrs.get("strides", kernel), "strides")
        padding = _pair(attrs.get("padding", 0), "padding")
        return one(
            values[0].dtype,
            _pool_output_shape(
                values[0].shape, kernel=kernel, stride=stride, padding=padding
            ),
        )

    if operation == Op.GLOBAL_AVG_POOL:
        _require_arity(operation, values, 1, 1)
        if values[0].rank != 4:
            raise ModelIRError("global_avg_pool expects rank-4 NCHW input")
        n, c, _, _ = values[0].shape
        return one(values[0].dtype, (n, c, 1, 1))

    if operation == Op.EMBEDDING_GATHER:
        _require_arity(operation, values, 2, 2)
        indices, weight = values
        if weight.rank != 2:
            raise ModelIRError("embedding_gather weight must be rank 2")
        return one(weight.dtype, (*indices.shape, weight.shape[1]))

    if operation == Op.ARGMAX:
        _require_arity(operation, values, 1, 1)
        input_value = values[0]
        axis = int(attrs.get("axis", -1))
        axis = axis + input_value.rank if axis < 0 else axis
        if axis < 0 or axis >= input_value.rank:
            raise ModelIRError("argmax axis out of range")
        shape = input_value.shape[:axis] + input_value.shape[axis + 1 :]
        return one(DType.I32, shape)

    if operation == Op.BATCH_NORM:
        _require_arity(operation, values, 5, 5)
        return one(values[0].dtype, values[0].shape)

    if operation == Op.GROUP_NORM:
        _require_arity(operation, values, 3, 3)
        return one(values[0].dtype, values[0].shape)

    if operation == Op.LAYER_NORM:
        _require_arity(operation, values, 3, 3)
        return one(values[0].dtype, values[0].shape)

    if operation == Op.RMS_NORM:
        _require_arity(operation, values, 2, 2)
        return one(values[0].dtype, values[0].shape)

    if operation in {
        Op.SCALED_DOT_PRODUCT_ATTENTION,
        Op.MASKED_ATTENTION,
    }:
        _require_arity(operation, values, 3, 4)
        q, k, v = values[0], values[1], values[2]
        if q.rank != 4 or k.rank != 4 or v.rank != 4:
            raise ModelIRError(f"{operation.value} expects rank-4 Q/K/V tensors")
        if q.shape[-1] != k.shape[-1]:
            raise ModelIRError("attention Q and K head dims do not match")
        if k.shape[-2] != v.shape[-2]:
            raise ModelIRError("attention K and V sequence lengths do not match")
        if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
            raise ModelIRError("attention batch/head dimensions do not match")
        return one(q.dtype, q.shape)

    if operation == Op.KV_CACHE_APPEND:
        _require_arity(operation, values, 2, 2)
        cache, value = values
        if cache.rank != value.rank or cache.shape[:-1] != value.shape[:-1]:
            raise ModelIRError(
                "kv_cache_append cache and value batch dims do not match"
            )
        if cache.shape[-1] != value.shape[-1]:
            raise ModelIRError("kv_cache_append head dims do not match")
        return one(cache.dtype, cache.shape)

    if operation == Op.KV_CACHE_GATHER:
        _require_arity(operation, values, 2, 2)
        cache, positions = values
        if positions.rank != 2:
            raise ModelIRError("kv_cache_gather positions must be rank 2")
        if cache.rank != 4:
            raise ModelIRError("kv_cache_gather cache must be rank 4")
        batch, heads, _, dim = cache.shape
        seq = positions.shape[1]
        return one(cache.dtype, (batch, heads, seq, dim))

    if operation == Op.CAUSAL_MASK:
        _require_arity(operation, values, 0, 0)
        if "shape" in attrs:
            return one(
                _as_dtype(attrs.get("dtype", DType.F32)),
                _as_shape(attrs["shape"]),
            )
        batch = int(attrs.get("batch", 1))
        query_len = int(attrs.get("query_len"))
        key_len = int(attrs.get("key_len"))
        return one(
            _as_dtype(attrs.get("dtype", DType.F32)),
            (batch, 1, query_len, key_len),
        )

    if operation == Op.RESHAPE:
        _require_arity(operation, values, 1, 1)
        requested = list(attrs.get("shape", ()))
        if requested.count(-1) > 1:
            raise ModelIRError("reshape allows at most one -1 dimension")
        if -1 in requested:
            known = math.prod(dim for dim in requested if dim != -1)
            if known == 0 or values[0].numel % known != 0:
                raise ModelIRError("cannot infer reshape -1 dimension")
            requested[requested.index(-1)] = values[0].numel // known
        shape = _as_shape(requested)
        if math.prod(shape) != values[0].numel:
            raise ModelIRError("reshape changes the number of elements")
        return one(values[0].dtype, shape)

    if operation == Op.TRANSPOSE:
        _require_arity(operation, values, 1, 1)
        rank = values[0].rank
        perm = tuple(
            int(axis) for axis in attrs.get("perm", tuple(reversed(range(rank))))
        )
        if sorted(perm) != list(range(rank)):
            raise ModelIRError("transpose perm must be a permutation of input axes")
        return one(values[0].dtype, tuple(values[0].shape[axis] for axis in perm))

    if operation == Op.CONCAT:
        if not values:
            raise ModelIRError("concat requires at least one input")
        axis = int(attrs.get("axis", 0))
        if axis < 0:
            axis += values[0].rank
        if axis < 0 or axis >= values[0].rank:
            raise ModelIRError("concat axis out of range")
        rank = values[0].rank
        reference = values[0].shape
        for value in values[1:]:
            if value.rank != rank:
                raise ModelIRError("concat inputs must have the same rank")
            for dim in range(rank):
                if dim != axis and value.shape[dim] != reference[dim]:
                    raise ModelIRError("concat non-axis dimensions do not match")
        output_axis = sum(value.shape[axis] for value in values)
        shape = reference[:axis] + (output_axis,) + reference[axis + 1 :]
        return one(_promote_dtype(*[value.dtype for value in values]), shape)

    if operation == Op.SLICE:
        _require_arity(operation, values, 1, 1)
        rank = values[0].rank
        starts = tuple(int(x) for x in attrs.get("starts", (0,) * rank))
        ends = tuple(int(x) for x in attrs.get("ends", values[0].shape))
        axes = tuple(int(x) for x in attrs.get("axes", tuple(range(rank))))
        steps = tuple(int(x) for x in attrs.get("steps", (1,) * rank))
        if not (len(starts) == len(ends) == len(axes) == len(steps)):
            raise ModelIRError("slice starts/ends/axes/steps lengths differ")
        dims = list(values[0].shape)
        for start, end, axis, step in zip(starts, ends, axes, steps):
            axis = axis + rank if axis < 0 else axis
            if axis < 0 or axis >= rank:
                raise ModelIRError("slice axis out of range")
            if step == 0:
                raise ModelIRError("slice step must be non-zero")
            dims[axis] = len(range(start, end, step))
        return one(values[0].dtype, tuple(dims))

    if operation == Op.PAD:
        _require_arity(operation, values, 1, 1)
        rank = values[0].rank
        pads = list(attrs.get("pads", ((0, 0),) * rank))
        if len(pads) != rank:
            raise ModelIRError("pad length must match input rank")
        dims = []
        for dim, pad in zip(values[0].shape, pads):
            before, after = int(pad[0]), int(pad[1])
            if before < 0 or after < 0:
                raise ModelIRError("pad values must be non-negative")
            dims.append(dim + before + after)
        return one(values[0].dtype, tuple(dims))

    if operation == Op.RESIZE:
        _require_arity(operation, values, 1, 1)
        if "sizes" in attrs:
            return one(values[0].dtype, _as_shape(attrs["sizes"]))
        if "scales" in attrs:
            scales = tuple(float(scale) for scale in attrs["scales"])
            if len(scales) != values[0].rank:
                raise ModelIRError("resize scales length must match input rank")
            return one(
                values[0].dtype,
                tuple(
                    max(int(dim * scale), 1)
                    for dim, scale in zip(values[0].shape, scales)
                ),
            )
        raise ModelIRError("resize requires sizes or scales")

    if operation == Op.EXPAND:
        _require_arity(operation, values, 1, 1)
        requested = list(attrs.get("shape", ()))
        if -1 in requested:
            rank = max(len(requested), values[0].rank)
            padded_input = (1,) * (rank - values[0].rank) + values[0].shape
            requested = [
                padded_input[index] if dim == -1 else dim
                for index, dim in enumerate(requested)
            ]
        output = _broadcast_shapes(values[0].shape, _as_shape(requested))
        return one(values[0].dtype, output)

    if operation == Op.SQUEEZE:
        _require_arity(operation, values, 1, 1)
        axes = (
            None if "axes" not in attrs else tuple(int(axis) for axis in attrs["axes"])
        )
        if axes is None:
            axes = tuple(axis for axis, dim in enumerate(values[0].shape) if dim == 1)
        axes = _normalize_axes(axes, values[0].rank)
        if any(values[0].shape[axis] != 1 for axis in axes):
            raise ModelIRError("squeeze axis must have size 1")
        return one(
            values[0].dtype,
            tuple(dim for axis, dim in enumerate(values[0].shape) if axis not in axes),
        )

    if operation == Op.UNSQUEEZE:
        _require_arity(operation, values, 1, 1)
        axes = tuple(int(axis) for axis in attrs.get("axes", ()))
        output_rank = values[0].rank + len(axes)
        normalized = sorted(axis + output_rank if axis < 0 else axis for axis in axes)
        if len(set(normalized)) != len(normalized) or any(
            axis < 0 or axis >= output_rank for axis in normalized
        ):
            raise ModelIRError("unsqueeze axes are out of range or duplicate")
        dims = list(values[0].shape)
        for axis in normalized:
            dims.insert(axis, 1)
        return one(values[0].dtype, tuple(dims))

    if operation == Op.BOX_DECODE:
        _require_arity(operation, values, 2, 2)
        return one(_promote_dtype(values[0].dtype, values[1].dtype), values[0].shape)

    if operation == Op.NMS:
        _require_arity(operation, values, 2, 2)
        if values[0].rank != 2:
            raise ModelIRError("nms boxes must be rank 2")
        return one(DType.I64, (values[0].shape[0],))

    raise ModelIRError(f"shape inference not implemented for op: {operation.value}")


def _require_arity(op: Op, values: list[Value], minimum: int, maximum: int) -> None:
    if not minimum <= len(values) <= maximum:
        expected = str(minimum) if minimum == maximum else f"{minimum}..{maximum}"
        raise ModelIRError(f"{op.value} expects {expected} inputs, got {len(values)}")


def _source_suffix(source: SourceLocation | None) -> str:
    return f" {source.format()}" if source else ""


def _validate_graph(graph: Graph) -> None:
    if not graph.name:
        raise ModelIRError("graph name must not be empty")

    defined: dict[str, Value] = {}
    for value in graph.inputs:
        if value.name in defined:
            raise ModelIRError(f"duplicate value name: {value.name}")
        defined[value.name] = value

    node_names: set[str] = set()
    for node in graph.nodes:
        if node.name in node_names:
            raise ModelIRError(f"duplicate node name: {node.name}")
        node_names.add(node.name)

        for value in node.inputs:
            if value.name not in defined:
                raise ModelIRError(
                    f"node {node.name} reads undefined value {value.name}"
                    f"{_source_suffix(node.source)}"
                )

        expected = infer_node_outputs(node.op, node.inputs, node.attrs)
        if len(node.outputs) != len(expected):
            raise ModelIRError(
                f"node {node.name} has {len(node.outputs)} outputs, "
                f"expected {len(expected)}"
                f"{_source_suffix(node.source)}"
            )

        for value, (dtype, shape) in zip(node.outputs, expected):
            if value.name in defined:
                raise ModelIRError(f"duplicate value name: {value.name}")
            if value.dtype != dtype or value.shape != shape:
                raise ModelIRError(
                    f"node {node.name} output {value.name} has "
                    f"({value.dtype.value}, {value.shape}), "
                    f"expected ({dtype.value}, {shape})"
                    f"{_source_suffix(node.source)}"
                )
            defined[value.name] = value

    for value in graph.outputs:
        if value.name not in defined:
            raise ModelIRError(f"graph output {value.name} is not defined")


def format_graph(graph: Graph) -> str:
    """Return a stable, human-readable textual form of ``graph``."""

    def value_text(value: Value) -> str:
        return f"%{value.name}: {value.dtype.value}<{'x'.join(map(str, value.shape))}>"

    def attrs_text(attrs: dict[str, Any]) -> str:
        if not attrs:
            return ""
        rendered = ", ".join(f"{key}={value!r}" for key, value in attrs.items())
        return f" {{{rendered}}}"

    lines = [f"graph @{graph.name}("]
    lines.extend(f"  {value_text(value)}" for value in graph.inputs)
    lines.append(") {")

    for node in graph.nodes:
        inputs = ", ".join(f"%{value.name}" for value in node.inputs)
        outputs = ", ".join(f"%{value.name}" for value in node.outputs)
        lines.append(
            f'  {outputs} = "{node.op.value}"({inputs}){attrs_text(node.attrs)}'
        )

    outputs = ", ".join(f"%{value.name}" for value in graph.outputs)
    lines.append(f"  return {outputs}")
    lines.append("}")
    return "\n".join(lines)
