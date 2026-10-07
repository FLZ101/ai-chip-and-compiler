"""Import a small ONNX CNN subset into the unified Model IR."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import numpy_helper

from ..mlir.importer import ONNXImportResult, ONNXImporter
from ..model_ir import DType, Graph
from .manual_frontend import ManualFrontend

_ONNX_DTYPE_MAP = {
    1: DType.F32,
    6: DType.I32,
    7: DType.I64,
    10: DType.F16,
    11: DType.F32,
    16: DType.BF16,
}


class ONNXFrontendError(Exception):
    """Raised when an ONNX model cannot be imported."""


class ONNXFrontend:
    """Import a supported ONNX graph into :class:`Graph`."""

    def __init__(self, name: str = "onnx_model"):
        self.name = name

    def convert(self, model: onnx.ModelProto) -> Graph:
        if model.graph is None:
            raise ONNXFrontendError("ONNX model has no graph")

        frontend = ManualFrontend(self.name)
        values: dict[str, object] = {}

        for initializer in model.graph.initializer:
            array = numpy_helper.to_array(initializer)
            dtype = self._array_dtype(array)
            values[initializer.name] = frontend.add_input(
                initializer.name, dtype, tuple(array.shape)
            )

        for graph_input in model.graph.input:
            if graph_input.name in values:
                continue
            dtype = self._elem_dtype(graph_input.type)
            shape = self._shape(graph_input.type)
            values[graph_input.name] = frontend.add_input(
                graph_input.name, dtype, shape
            )

        for node in model.graph.node:
            self._convert_node(frontend, node, values)

        output_values = []
        for output in model.graph.output:
            if output.name not in values:
                raise ONNXFrontendError(f"ONNX output {output.name!r} is not defined")
            output_values.append(values[output.name])
        frontend.output(*output_values)
        return frontend.build()

    def import_mlir(self, model: onnx.ModelProto) -> ONNXImportResult:
        """Import an ONNX model through the real torch-mlir ONNX importer.

        This is the Phase 2.2 entry point.  It avoids the manual per-op
        :meth:`convert` path and returns parsed MLIR plus the ONNX
        initializer/dynamic-input metadata needed by later lowering passes.
        """
        return ONNXImporter().import_model(model)

    def _convert_node(
        self,
        frontend: ManualFrontend,
        node: onnx.NodeProto,
        values: dict[str, object],
    ) -> None:
        op = node.op_type
        if op == "Conv":
            values[node.output[0]] = self._convert_conv(frontend, node, values)
            return
        if op == "Relu":
            values[node.output[0]] = frontend.relu(values[node.input[0]])
            return
        if op == "MaxPool":
            values[node.output[0]] = self._convert_max_pool(frontend, node, values)
            return
        if op == "Add":
            values[node.output[0]] = frontend.add(
                values[node.input[0]], values[node.input[1]]
            )
            return
        if op == "Flatten":
            values[node.output[0]] = self._convert_flatten(frontend, node, values)
            return
        raise ONNXFrontendError(f"unsupported ONNX op: {op}")

    def _convert_conv(
        self,
        frontend: ManualFrontend,
        node: onnx.NodeProto,
        values: dict[str, object],
    ) -> object:
        strides = _attr_ints(node, "strides", [1, 1])
        pads = _attr_ints(node, "pads", [0, 0, 0, 0])
        dilations = _attr_ints(node, "dilations", [1, 1])
        group = _attr_int(node, "group", 1)
        if len(pads) == 4 and pads[0] != pads[2] or pads[1] != pads[3]:
            raise ONNXFrontendError("ONNX conv requires symmetric padding")
        padding = (pads[0], pads[1])
        bias = values.get(node.input[2]) if len(node.input) > 2 else None
        return frontend.conv2d(
            values[node.input[0]],
            values[node.input[1]],
            bias,
            strides=(strides[0], strides[1]),
            padding=padding,
            dilation=(dilations[0], dilations[1]),
            groups=group,
        )

    def _convert_max_pool(
        self,
        frontend: ManualFrontend,
        node: onnx.NodeProto,
        values: dict[str, object],
    ) -> object:
        kernel = _attr_ints(node, "kernel_shape", [2, 2])
        strides = _attr_ints(node, "strides", kernel)
        pads = _attr_ints(node, "pads", [0, 0, 0, 0])
        if any(pad != 0 for pad in pads):
            raise ONNXFrontendError("ONNX maxpool padding is not supported yet")
        return frontend.max_pool2d(
            values[node.input[0]],
            kernel=(kernel[0], kernel[1]),
            strides=(strides[0], strides[1]),
        )

    def _convert_flatten(
        self,
        frontend: ManualFrontend,
        node: onnx.NodeProto,
        values: dict[str, object],
    ) -> object:
        value = values[node.input[0]]
        axis = _attr_int(node, "axis", 1)
        if axis < 0:
            axis += len(value.shape)
        leading = 1
        for dim in value.shape[:axis]:
            leading *= dim
        return frontend.reshape(value, (leading, -1))

    def _elem_dtype(self, type_proto: onnx.TypeProto) -> DType:
        tensor_type = type_proto.tensor_type
        if tensor_type is None or not tensor_type.HasField("elem_type"):
            raise ONNXFrontendError("ONNX value is not a tensor")
        try:
            return _ONNX_DTYPE_MAP[tensor_type.elem_type]
        except KeyError as exc:
            raise ONNXFrontendError(
                f"unsupported ONNX dtype: {tensor_type.elem_type}"
            ) from exc

    def _shape(self, type_proto: onnx.TypeProto) -> tuple[int, ...]:
        tensor_type = type_proto.tensor_type
        dims = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(dim.dim_value)
            else:
                raise ONNXFrontendError("dynamic ONNX dimensions are not supported")
        return tuple(dims)

    def _array_dtype(self, array: np.ndarray) -> DType:
        if array.dtype == np.float32:
            return DType.F32
        if array.dtype == np.float64:
            return DType.F32
        if array.dtype == np.int64:
            return DType.I64
        if array.dtype == np.int32:
            return DType.I32
        raise ONNXFrontendError(f"unsupported ONNX initializer dtype: {array.dtype}")


def _attr_ints(node: onnx.NodeProto, name: str, default: list[int]) -> list[int]:
    for attr in node.attribute:
        if attr.name == name:
            return [int(value) for value in attr.ints]
    return default


def _attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default
