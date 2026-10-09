"""Manual Python builder for the unified Model IR.

This frontend is intentionally explicit.  It exists so operator tests and
small CNN/Transformer graphs can be constructed before the ONNX and PyTorch
importers are added.  Every method builds ordinary :class:`~compiler.model_ir.Node`
objects and relies on the shared shape inference in
:mod:`compiler.model_ir`.
"""

from __future__ import annotations

from typing import Any

from ..model_ir import DType, Graph, Op, Value


class ManualFrontend:
    """Build a :class:`Graph` with a small, typed Python API."""

    def __init__(self, name: str = "model"):
        self.graph = Graph(name=name)

    def add_input(
        self,
        name: str,
        dtype: DType | str,
        shape: tuple[int, ...],
    ) -> Value:
        return self.graph.add_input(name, dtype, tuple(shape))

    def constant(
        self,
        value: Any,
        *,
        dtype: DType | str = DType.F32,
        name: str | None = None,
    ) -> Value:
        return self._emit(
            Op.CONSTANT, [], {"value": value, "dtype": DType(dtype)}, name=name
        )

    def add(self, a: Value, b: Value, *, name: str | None = None) -> Value:
        return self._emit(Op.ADD, [a, b], name=name)

    def sub(self, a: Value, b: Value, *, name: str | None = None) -> Value:
        return self._emit(Op.SUB, [a, b], name=name)

    def mul(self, a: Value, b: Value, *, name: str | None = None) -> Value:
        return self._emit(Op.MUL, [a, b], name=name)

    def div(self, a: Value, b: Value, *, name: str | None = None) -> Value:
        return self._emit(Op.DIV, [a, b], name=name)

    def fma(
        self,
        a: Value,
        b: Value,
        c: Value,
        *,
        name: str | None = None,
    ) -> Value:
        return self._emit(Op.FMA, [a, b, c], name=name)

    def maximum(self, a: Value, b: Value, *, name: str | None = None) -> Value:
        return self._emit(Op.MAX, [a, b], name=name)

    def minimum(self, a: Value, b: Value, *, name: str | None = None) -> Value:
        return self._emit(Op.MIN, [a, b], name=name)

    def relu(self, x: Value, *, name: str | None = None) -> Value:
        return self._emit(Op.RELU, [x], name=name)

    def leaky_relu(
        self,
        x: Value,
        *,
        alpha: float = 0.01,
        name: str | None = None,
    ) -> Value:
        return self._emit(Op.LEAKY_RELU, [x], {"alpha": alpha}, name=name)

    def silu(self, x: Value, *, name: str | None = None) -> Value:
        return self._emit(Op.SILU, [x], name=name)

    def gelu(self, x: Value, *, name: str | None = None) -> Value:
        return self._emit(Op.GELU, [x], name=name)

    def sigmoid(self, x: Value, *, name: str | None = None) -> Value:
        return self._emit(Op.SIGMOID, [x], name=name)

    def hard_sigmoid(self, x: Value, *, name: str | None = None) -> Value:
        return self._emit(Op.HARD_SIGMOID, [x], name=name)

    def reduce_sum(
        self,
        x: Value,
        *,
        axes: tuple[int, ...] | None = None,
        keepdims: bool = False,
        name: str | None = None,
    ) -> Value:
        return self._emit(
            Op.REDUCE_SUM, [x], {"axes": axes, "keepdims": keepdims}, name=name
        )

    def reduce_mean(
        self,
        x: Value,
        *,
        axes: tuple[int, ...] | None = None,
        keepdims: bool = False,
        name: str | None = None,
    ) -> Value:
        return self._emit(
            Op.REDUCE_MEAN, [x], {"axes": axes, "keepdims": keepdims}, name=name
        )

    def reduce_max(
        self,
        x: Value,
        *,
        axes: tuple[int, ...] | None = None,
        keepdims: bool = False,
        name: str | None = None,
    ) -> Value:
        return self._emit(
            Op.REDUCE_MAX, [x], {"axes": axes, "keepdims": keepdims}, name=name
        )

    def matmul(self, a: Value, b: Value, *, name: str | None = None) -> Value:
        return self._emit(Op.MATMUL, [a, b], name=name)

    def batch_matmul(self, a: Value, b: Value, *, name: str | None = None) -> Value:
        return self._emit(Op.BATCH_MATMUL, [a, b], name=name)

    def linear(
        self,
        x: Value,
        weight: Value,
        bias: Value | None = None,
        *,
        name: str | None = None,
    ) -> Value:
        inputs = [x, weight] if bias is None else [x, weight, bias]
        return self._emit(Op.LINEAR, inputs, name=name)

    def conv2d(
        self,
        x: Value,
        weight: Value,
        bias: Value | None = None,
        *,
        strides: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        name: str | None = None,
    ) -> Value:
        inputs = [x, weight] if bias is None else [x, weight, bias]
        return self._emit(
            Op.CONV2D,
            inputs,
            {
                "strides": strides,
                "padding": padding,
                "dilation": dilation,
                "groups": groups,
            },
            name=name,
        )

    def depthwise_conv2d(
        self,
        x: Value,
        weight: Value,
        bias: Value | None = None,
        *,
        strides: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        name: str | None = None,
    ) -> Value:
        inputs = [x, weight] if bias is None else [x, weight, bias]
        return self._emit(
            Op.DEPTHWISE_CONV2D,
            inputs,
            {"strides": strides, "padding": padding, "dilation": dilation},
            name=name,
        )

    def max_pool2d(
        self,
        x: Value,
        *,
        kernel: int | tuple[int, int] = 2,
        strides: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        name: str | None = None,
    ) -> Value:
        attrs: dict[str, Any] = {"kernel": kernel, "padding": padding}
        if strides is not None:
            attrs["strides"] = strides
        return self._emit(Op.MAX_POOL2D, [x], attrs, name=name)

    def avg_pool2d(
        self,
        x: Value,
        *,
        kernel: int | tuple[int, int] = 2,
        strides: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        name: str | None = None,
    ) -> Value:
        attrs: dict[str, Any] = {"kernel": kernel, "padding": padding}
        if strides is not None:
            attrs["strides"] = strides
        return self._emit(Op.AVG_POOL2D, [x], attrs, name=name)

    def global_avg_pool(self, x: Value, *, name: str | None = None) -> Value:
        return self._emit(Op.GLOBAL_AVG_POOL, [x], name=name)

    def embedding_gather(
        self,
        indices: Value,
        weight: Value,
        *,
        name: str | None = None,
    ) -> Value:
        return self._emit(Op.EMBEDDING_GATHER, [indices, weight], name=name)

    def argmax(
        self,
        x: Value,
        *,
        axis: int = -1,
        name: str | None = None,
    ) -> Value:
        return self._emit(Op.ARGMAX, [x], {"axis": axis}, name=name)

    def batch_norm(
        self,
        x: Value,
        scale: Value,
        bias: Value,
        mean: Value,
        variance: Value,
        *,
        epsilon: float = 1e-5,
        name: str | None = None,
    ) -> Value:
        return self._emit(
            Op.BATCH_NORM,
            [x, scale, bias, mean, variance],
            {"epsilon": epsilon},
            name=name,
        )

    def group_norm(
        self,
        x: Value,
        weight: Value,
        bias: Value,
        *,
        num_groups: int,
        epsilon: float = 1e-5,
        name: str | None = None,
    ) -> Value:
        return self._emit(
            Op.GROUP_NORM,
            [x, weight, bias],
            {"num_groups": num_groups, "epsilon": epsilon},
            name=name,
        )

    def layer_norm(
        self,
        x: Value,
        weight: Value,
        bias: Value,
        *,
        epsilon: float = 1e-5,
        name: str | None = None,
    ) -> Value:
        return self._emit(
            Op.LAYER_NORM, [x, weight, bias], {"epsilon": epsilon}, name=name
        )

    def rms_norm(
        self,
        x: Value,
        weight: Value,
        *,
        epsilon: float = 1e-5,
        name: str | None = None,
    ) -> Value:
        return self._emit(Op.RMS_NORM, [x, weight], {"epsilon": epsilon}, name=name)

    def softmax(
        self,
        x: Value,
        *,
        axis: int = -1,
        name: str | None = None,
    ) -> Value:
        return self._emit(Op.SOFTMAX, [x], {"axis": axis}, name=name)

    def masked_softmax(
        self,
        logits: Value,
        mask: Value,
        *,
        axis: int = -1,
        name: str | None = None,
    ) -> Value:
        return self._emit(Op.MASKED_SOFTMAX, [logits, mask], {"axis": axis}, name=name)

    def scaled_dot_product_attention(
        self,
        q: Value,
        k: Value,
        v: Value,
        *,
        scale: float | None = None,
        name: str | None = None,
    ) -> Value:
        attrs: dict[str, Any] = {}
        if scale is not None:
            attrs["scale"] = scale
        return self._emit(Op.SCALED_DOT_PRODUCT_ATTENTION, [q, k, v], attrs, name=name)

    def masked_attention(
        self,
        q: Value,
        k: Value,
        v: Value,
        mask: Value,
        *,
        scale: float | None = None,
        name: str | None = None,
    ) -> Value:
        attrs: dict[str, Any] = {}
        if scale is not None:
            attrs["scale"] = scale
        return self._emit(Op.MASKED_ATTENTION, [q, k, v, mask], attrs, name=name)

    def rope(
        self,
        x: Value,
        cos: Value,
        sin: Value,
        *,
        name: str | None = None,
    ) -> Value:
        return self._emit(Op.ROPE, [x, cos, sin], name=name)

    def kv_cache_append(
        self,
        cache: Value,
        value: Value,
        *,
        name: str | None = None,
    ) -> Value:
        return self._emit(Op.KV_CACHE_APPEND, [cache, value], name=name)

    def kv_cache_gather(
        self,
        cache: Value,
        positions: Value,
        *,
        name: str | None = None,
    ) -> Value:
        return self._emit(Op.KV_CACHE_GATHER, [cache, positions], name=name)

    def causal_mask(
        self,
        *,
        query_len: int,
        key_len: int,
        batch: int = 1,
        dtype: DType | str = DType.F32,
        name: str | None = None,
    ) -> Value:
        return self._emit(
            Op.CAUSAL_MASK,
            [],
            {
                "batch": batch,
                "query_len": query_len,
                "key_len": key_len,
                "dtype": DType(dtype),
            },
            name=name,
        )

    def reshape(
        self, x: Value, shape: tuple[int, ...], *, name: str | None = None
    ) -> Value:
        return self._emit(Op.RESHAPE, [x], {"shape": tuple(shape)}, name=name)

    def transpose(
        self,
        x: Value,
        perm: tuple[int, ...] | None = None,
        *,
        name: str | None = None,
    ) -> Value:
        attrs: dict[str, Any] = {}
        if perm is not None:
            attrs["perm"] = tuple(perm)
        return self._emit(Op.TRANSPOSE, [x], attrs, name=name)

    def concat(self, *inputs: Value, axis: int = 0, name: str | None = None) -> Value:
        return self._emit(Op.CONCAT, list(inputs), {"axis": axis}, name=name)

    def slice(
        self,
        x: Value,
        starts: tuple[int, ...],
        ends: tuple[int, ...],
        *,
        axes: tuple[int, ...] | None = None,
        steps: tuple[int, ...] | None = None,
        name: str | None = None,
    ) -> Value:
        axes = tuple(range(x.rank)) if axes is None else axes
        steps = (1,) * len(axes) if steps is None else steps
        return self._emit(
            Op.SLICE,
            [x],
            {"starts": starts, "ends": ends, "axes": axes, "steps": steps},
            name=name,
        )

    def pad(
        self,
        x: Value,
        pads: tuple[tuple[int, int], ...],
        *,
        value: float = 0.0,
        name: str | None = None,
    ) -> Value:
        return self._emit(Op.PAD, [x], {"pads": pads, "value": value}, name=name)

    def resize(
        self,
        x: Value,
        *,
        sizes: tuple[int, ...] | None = None,
        scales: tuple[float, ...] | None = None,
        name: str | None = None,
    ) -> Value:
        attrs: dict[str, Any] = {}
        if sizes is not None:
            attrs["sizes"] = tuple(sizes)
        if scales is not None:
            attrs["scales"] = tuple(scales)
        return self._emit(Op.RESIZE, [x], attrs, name=name)

    def expand(
        self, x: Value, shape: tuple[int, ...], *, name: str | None = None
    ) -> Value:
        return self._emit(Op.EXPAND, [x], {"shape": tuple(shape)}, name=name)

    def squeeze(
        self,
        x: Value,
        axes: tuple[int, ...] | None = None,
        *,
        name: str | None = None,
    ) -> Value:
        attrs: dict[str, Any] = {}
        if axes is not None:
            attrs["axes"] = tuple(axes)
        return self._emit(Op.SQUEEZE, [x], attrs, name=name)

    def unsqueeze(
        self, x: Value, axes: tuple[int, ...], *, name: str | None = None
    ) -> Value:
        return self._emit(Op.UNSQUEEZE, [x], {"axes": tuple(axes)}, name=name)

    def box_decode(
        self, boxes: Value, deltas: Value, *, name: str | None = None
    ) -> Value:
        return self._emit(Op.BOX_DECODE, [boxes, deltas], name=name)

    def nms(
        self,
        boxes: Value,
        scores: Value,
        *,
        iou_threshold: float = 0.5,
        max_output: int = 0,
        name: str | None = None,
    ) -> Value:
        return self._emit(
            Op.NMS,
            [boxes, scores],
            {"iou_threshold": iou_threshold, "max_output": max_output},
            name=name,
        )

    def output(self, *values: Value) -> None:
        self.graph.mark_outputs(values)

    def build(self) -> Graph:
        self.graph.validate()
        return self.graph

    def _emit(
        self,
        op: Op,
        inputs: list[Value],
        attrs: dict[str, Any] | None = None,
        *,
        name: str | None = None,
    ) -> Value:
        outputs = self.graph.add_node(op, inputs, attrs, name=name)
        if len(outputs) != 1:
            raise ValueError(f"{op.value} did not produce exactly one output")
        return outputs[0]
