"""Import a small PyTorch FX Transformer subset into the unified Model IR."""

from __future__ import annotations

import operator
from typing import Any

import torch
import torch.nn as nn

from ..model_ir import DType, Graph
from ..mlir.importer import TorchMLIRImporter, TorchMLIRImportResult
from .attention_functions import causal_mask, masked_attention
from .detection_functions import box_decode, nms
from .manual_frontend import ManualFrontend


class TorchFXFrontendError(Exception):
    """Raised when an FX graph cannot be imported."""


class TorchFXFrontend:
    """Import a supported ``torch.fx.GraphModule`` into :class:`Graph`."""

    def __init__(
        self,
        name: str = "fx_model",
        *,
        input_shapes: dict[str, tuple[tuple[int, ...], DType | str]] | None = None,
    ):
        self.name = name
        self.input_shapes = input_shapes or {}
        self._attr_names: dict[str, str] = {}

    def convert(self, graph_module: torch.fx.GraphModule) -> Graph:
        frontend = ManualFrontend(self.name)
        values: dict[str, object] = {}

        for node in graph_module.graph.nodes:
            if node.op == "placeholder":
                shape, dtype = self._placeholder_spec(node.name)
                values[node.name] = frontend.add_input(node.name, dtype, shape)
            elif node.op == "get_attr":
                value = self._resolve_attr(graph_module, node.target)
                shape = tuple(value.shape)
                dtype = self._tensor_dtype(value)
                values[node.name] = frontend.add_input(node.name, dtype, shape)
                self._attr_names[node.target] = node.name
            elif node.op == "call_function":
                values[node.name] = self._convert_call_function(frontend, node, values)
            elif node.op == "call_module":
                values[node.name] = self._convert_call_module(
                    frontend, graph_module, node, values
                )
            elif node.op == "call_method":
                values[node.name] = self._convert_call_method(frontend, node, values)
            elif node.op == "output":
                outputs = [self._resolve_arg(arg, values) for arg in node.args]
                frontend.output(*outputs)
            else:
                raise TorchFXFrontendError(f"unsupported FX node op: {node.op}")

        return frontend.build()

    def import_mlir(
        self,
        module: torch.nn.Module,
        *example_args: Any,
        example_kwargs: dict[str, Any] | None = None,
        func_name: str | None = None,
        dynamic_shapes: dict[str, Any] | tuple[Any, ...] | None = None,
        strict: bool = False,
    ) -> TorchMLIRImportResult:
        """Import a real PyTorch module through the ``torch-mlir`` FX path.

        This is the Phase 2.1 entry point.  It keeps the legacy manual
        :meth:`convert` path for simulator E2E tests while making the
        production MLIR import available alongside it.  If no example inputs
        are supplied, a single input declared in ``input_shapes`` is used.
        """
        if not example_args and not example_kwargs and self.input_shapes:
            example_args = self._example_args_from_single_shape()

        importer = TorchMLIRImporter()
        return importer.import_module(
            module,
            *example_args,
            example_kwargs=example_kwargs,
            func_name=func_name or self.name,
            dynamic_shapes=dynamic_shapes,
            strict=strict,
        )

    def _example_args_from_single_shape(self) -> tuple[torch.Tensor, ...]:
        if len(self.input_shapes) != 1:
            raise TorchFXFrontendError(
                "torch-mlir import with multiple inputs requires explicit example_args"
            )
        ((_, (shape, dtype)),) = self.input_shapes.items()
        torch_dtype = _dtype_to_torch(DType(dtype))
        return (torch.zeros(tuple(shape), dtype=torch_dtype),)

    def _placeholder_spec(self, name: str) -> tuple[tuple[int, ...], DType]:
        if name not in self.input_shapes:
            raise TorchFXFrontendError(f"missing input shape for placeholder {name!r}")
        shape, dtype = self.input_shapes[name]
        return tuple(shape), DType(dtype)

    def _convert_call_function(
        self,
        frontend: ManualFrontend,
        node: torch.fx.Node,
        values: dict[str, object],
    ) -> object:
        target = node.target
        if target is torch.nn.functional.linear:
            x = self._resolve_arg(node.args[0], values)
            weight = self._resolve_arg(node.args[1], values)
            bias = (
                self._resolve_arg(node.args[2], values)
                if len(node.args) > 2 and node.args[2] is not None
                else None
            )
            return frontend.linear(x, weight, bias)
        if target is torch.nn.functional.layer_norm:
            x = self._resolve_arg(node.args[0], values)
            weight = self._resolve_arg(node.args[2], values)
            bias = self._resolve_arg(node.args[3], values)
            eps = node.kwargs.get("eps", 1e-5)
            return frontend.layer_norm(x, weight, bias, epsilon=eps)
        if target is torch.nn.functional.softmax:
            x = self._resolve_arg(node.args[0], values)
            axis = int(node.kwargs.get("dim", node.args[1]))
            return frontend.softmax(x, axis=axis)
        if target in {torch.relu, torch.nn.functional.relu}:
            return frontend.relu(self._resolve_arg(node.args[0], values))
        if target in {torch.add, operator.add}:
            return frontend.add(
                self._resolve_arg(node.args[0], values),
                self._resolve_arg(node.args[1], values),
            )
        if target is box_decode:
            boxes = self._resolve_arg(node.args[0], values)
            deltas = self._resolve_arg(node.args[1], values)
            return frontend.box_decode(boxes, deltas)
        if target is nms:
            boxes = self._resolve_arg(node.args[0], values)
            scores = self._resolve_arg(node.args[1], values)
            threshold = self._resolve_iou_threshold(node)
            return frontend.nms(boxes, scores, iou_threshold=threshold)
        if target is causal_mask:
            query = self._resolve_arg(node.args[0], values)
            key = self._resolve_arg(node.args[1], values)
            return frontend.causal_mask(
                query_len=query.shape[2],
                key_len=key.shape[2],
                batch=query.shape[0],
            )
        if target is masked_attention:
            q = self._resolve_arg(node.args[0], values)
            k = self._resolve_arg(node.args[1], values)
            v = self._resolve_arg(node.args[2], values)
            mask = self._resolve_arg(node.args[3], values)
            scale = self._resolve_optional_float(node, "scale", 4)
            return frontend.masked_attention(q, k, v, mask, scale=scale)
        raise TorchFXFrontendError(f"unsupported FX function: {target}")

    def _convert_call_module(
        self,
        frontend: ManualFrontend,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.Node,
        values: dict[str, object],
    ) -> object:
        module = self._resolve_attr(graph_module, node.target)
        if isinstance(module, nn.Linear):
            x = self._resolve_arg(node.args[0], values)
            weight = self._add_param_input(
                frontend, f"{node.target}_weight", module.weight
            )
            bias = None
            if module.bias is not None:
                bias = self._add_param_input(
                    frontend, f"{node.target}_bias", module.bias
                )
            return frontend.linear(x, weight, bias)
        if isinstance(module, nn.LayerNorm):
            x = self._resolve_arg(node.args[0], values)
            weight = self._add_param_input(
                frontend, f"{node.target}_weight", module.weight
            )
            bias = self._add_param_input(frontend, f"{node.target}_bias", module.bias)
            return frontend.layer_norm(
                x,
                weight,
                bias,
                epsilon=module.eps,
            )
        if isinstance(module, nn.ReLU):
            return frontend.relu(self._resolve_arg(node.args[0], values))
        if isinstance(module, nn.Conv2d):
            x = self._resolve_arg(node.args[0], values)
            weight = self._add_param_input(
                frontend, f"{node.target}_weight", module.weight
            )
            bias = None
            if module.bias is not None:
                bias = self._add_param_input(
                    frontend, f"{node.target}_bias", module.bias
                )
            return frontend.conv2d(
                x,
                weight,
                bias,
                strides=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
            )
        if isinstance(module, nn.MaxPool2d):
            x = self._resolve_arg(node.args[0], values)
            return frontend.max_pool2d(
                x,
                kernel=module.kernel_size,
                strides=module.stride,
                padding=module.padding,
            )
        raise TorchFXFrontendError(f"unsupported FX module: {type(module).__name__}")

    def _convert_call_method(
        self,
        frontend: ManualFrontend,
        node: torch.fx.Node,
        values: dict[str, object],
    ) -> object:
        value = self._resolve_arg(node.args[0], values)
        if node.target == "reshape":
            shape = tuple(node.args[1])
            return frontend.reshape(value, shape)
        if node.target in {"transpose", "permute"}:
            perm = tuple(node.args[1])
            return frontend.transpose(value, perm)
        raise TorchFXFrontendError(f"unsupported FX method: {node.target}")

    def _resolve_arg(self, arg: Any, values: dict[str, object]) -> object:
        if isinstance(arg, torch.fx.Node):
            if arg.name not in values:
                raise TorchFXFrontendError(f"undefined FX value: {arg.name}")
            return values[arg.name]
        if isinstance(arg, (tuple, list)):
            return tuple(self._resolve_arg(item, values) for item in arg)
        raise TorchFXFrontendError(f"unsupported FX argument: {arg!r}")

    @staticmethod
    def _resolve_iou_threshold(node: torch.fx.Node) -> float:
        if "iou_threshold" in node.kwargs:
            return float(node.kwargs["iou_threshold"])
        if len(node.args) > 2:
            return float(node.args[2])
        return 0.5

    @staticmethod
    def _resolve_optional_float(
        node: torch.fx.Node,
        key: str,
        positional_index: int,
    ) -> float | None:
        if key in node.kwargs and node.kwargs[key] is not None:
            return float(node.kwargs[key])
        if (
            len(node.args) > positional_index
            and node.args[positional_index] is not None
        ):
            return float(node.args[positional_index])
        return None

    def _resolve_attr(self, graph_module: torch.fx.GraphModule, target: str) -> Any:
        current: Any = graph_module
        for part in target.split("."):
            current = getattr(current, part)
        return current

    def _tensor_dtype(self, value: torch.Tensor) -> DType:
        if value.dtype.is_floating_point:
            return DType.F32
        if value.dtype == torch.int64:
            return DType.I64
        if value.dtype == torch.int32:
            return DType.I32
        raise TorchFXFrontendError(f"unsupported FX tensor dtype: {value.dtype}")

    def _add_param_input(
        self,
        frontend: ManualFrontend,
        name: str,
        tensor: torch.Tensor,
    ) -> object:
        value = tensor.detach().cpu()
        return frontend.add_input(name, self._tensor_dtype(value), tuple(value.shape))


def _dtype_to_torch(dtype: DType) -> torch.dtype:
    """Map the small compiler dtype enum to a PyTorch dtype."""
    mapping = {
        DType.F32: torch.float32,
        DType.F16: torch.float16,
        DType.BF16: torch.bfloat16,
        DType.I32: torch.int32,
        DType.I64: torch.int64,
        DType.I8: torch.int8,
        DType.U8: torch.uint8,
        DType.BOOL: torch.bool,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise TorchFXFrontendError(
            f"cannot synthesize example tensor for dtype {dtype.value!r}"
        ) from exc
