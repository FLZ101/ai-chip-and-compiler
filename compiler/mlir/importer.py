"""Import PyTorch FX programs with the real ``torch-mlir`` frontend.

Phase 2.1 replaces the earlier hand-written FX-to-Model-IR walker with the
production ``torch-mlir`` importer.  The importer in this module performs three
related jobs:

1. export an ``nn.Module`` to an :class:`torch.export.ExportedProgram`,
2. import that exported FX program into an actual MLIR ``Module`` with
   ``torch_mlir.fx.export_and_import``, and
3. record the lifted parameter/buffer/constant name mappings from the export
   signature so later lowering and diagnostics can refer to the original
   PyTorch names.

The module depends on PyTorch and ``torch-mlir``, which are imported at the top
of the file because the production MLIR frontend is a first-class compiler
dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import onnx
import torch
from torch_mlir.dialects import torch as torch_dialect
from torch_mlir.extras import onnx_importer
from torch_mlir import fx as torch_mlir_fx
from torch_mlir.compiler_utils import OutputType
from torch_mlir.ir import Context


class TorchMLIRImportError(Exception):
    """Raised when a PyTorch program cannot be imported into MLIR."""


class ONNXImportError(Exception):
    """Raised when an ONNX model cannot be imported into MLIR."""


@dataclass(frozen=True)
class TorchMLIRImportResult:
    """A parsed MLIR module plus the metadata needed by later passes.

    Attributes:
        text: Stable, human-readable MLIR text for the imported module.
        function_name: The ``func.func`` symbol chosen for the entry point.
        parameter_map: Lifted placeholder name -> original PyTorch parameter name.
        buffer_map: Lifted placeholder name -> original PyTorch buffer name.
        constant_map: Lifted placeholder name -> original lifted constant name.
        mlir_module: The in-memory ``torch_mlir.ir.Module``.  It must be kept
            alive together with its context if callers need to inspect it.
    """

    text: str
    function_name: str
    mlir_module: Any = field(repr=False, compare=False)
    parameter_map: dict[str, str] = field(default_factory=dict)
    buffer_map: dict[str, str] = field(default_factory=dict)
    constant_map: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ONNXImportResult:
    """A parsed MLIR module produced by the real torch-mlir ONNX importer.

    Attributes:
        text: Stable MLIR text for the imported module.
        initializer_names: ONNX initializer names preserved from the model.
        dynamic_inputs: Non-initializer graph inputs mapped to their ONNX
            dimensions.  Static dimensions are integers, dynamic dimensions are
            the original ``dim_param`` strings, and unknown dimensions are
            ``None``.
        mlir_module: The in-memory ``torch_mlir.ir.Module``.
    """

    text: str
    initializer_names: tuple[str, ...]
    dynamic_inputs: dict[str, tuple[int | str | None, ...]]
    mlir_module: Any = field(repr=False, compare=False)


class TorchMLIRImporter:
    """Import an ``nn.Module`` or :class:`torch.export.ExportedProgram`.

    Example::

        importer = TorchMLIRImporter()
        result = importer.import_module(
            model,
            torch.zeros(1, 3, 224, 224),
            func_name="resnet_block",
        )
        print(result.text)
        print(result.parameter_map)
    """

    def import_module(
        self,
        module: Any,
        *example_args: Any,
        example_kwargs: dict[str, Any] | None = None,
        func_name: str = "main",
        dynamic_shapes: dict[str, Any] | tuple[Any, ...] | None = None,
        strict: bool = False,
        output_type: str = "raw",
        decomposition_table: dict[Any, Any] | None = None,
    ) -> TorchMLIRImportResult:
        """Export and import ``module`` into a real MLIR module.

        ``example_args`` and ``example_kwargs`` are passed to
        :func:`torch.export.export` when ``module`` is not already an
        :class:`~torch.export.ExportedProgram`.  Supplying concrete example
        inputs is the normal static-shape path required by the first real-model
        stage.
        """
        output_type_enum = OutputType.get(output_type)
        function_name = _sanitize_identifier(func_name)

        # ``export_and_import`` accepts both an ``nn.Module`` and an
        # ``ExportedProgram``.  If the caller already exported the module, the
        # export half is skipped and only the MLIR import half runs below.
        if isinstance(module, torch.export.ExportedProgram):
            exported_program = module
        else:
            if not example_args and not example_kwargs:
                raise TorchMLIRImportError(
                    "torch-mlir import requires example inputs for a PyTorch nn.Module"
                )
            # Export here instead of passing the raw module straight to
            # ``export_and_import`` because we need the graph signature to
            # preserve the original parameter/buffer/constant names.
            exported_program = torch.export.export(
                module,
                example_args,
                example_kwargs or {},
                dynamic_shapes=dynamic_shapes,
                strict=strict,
            )

        parameter_map = dict(
            getattr(exported_program.graph_signature, "inputs_to_parameters", {}) or {}
        )
        buffer_map = dict(
            getattr(exported_program.graph_signature, "inputs_to_buffers", {}) or {}
        )
        constant_map = dict(
            getattr(
                exported_program.graph_signature,
                "inputs_to_lifted_tensor_constants",
                {},
            )
            or {}
        )

        # The PyTorch export stage is already complete.  This call performs the
        # remaining MLIR import stage and returns a parsed ``ir.Module``.
        mlir_module = torch_mlir_fx.export_and_import(
            exported_program,
            output_type=output_type_enum,
            func_name=function_name,
            decomposition_table=decomposition_table,
        )
        text = str(mlir_module)

        return TorchMLIRImportResult(
            text=text,
            function_name=function_name,
            parameter_map=parameter_map,
            buffer_map=buffer_map,
            constant_map=constant_map,
            mlir_module=mlir_module,
        )


class ONNXImporter:
    """Import an ONNX model into a real MLIR module with torch-mlir."""

    def import_model(
        self,
        model: onnx.ModelProto,
        *,
        data_prop: bool = True,
        verify: bool = True,
    ) -> ONNXImportResult:
        """Import ``model`` without manual per-op ONNX mapping.

        ONNX shape inference is run first so dynamic dimensions and node result
        types are available to torch-mlir.  Initializers are handled by the
        importer's default ``elide_initialized_inputs`` behavior.
        """
        if model.graph is None:
            raise ONNXImportError("ONNX model has no graph")

        inferred_model = onnx.shape_inference.infer_shapes(
            model,
            data_prop=data_prop,
        )
        initializer_names = tuple(
            initializer.name for initializer in inferred_model.graph.initializer
        )
        dynamic_inputs = _onnx_dynamic_inputs(inferred_model.graph, initializer_names)

        context = Context()
        torch_dialect.register_dialect(context)
        config = onnx_importer.Config()
        model_info = onnx_importer.ModelInfo(inferred_model, config=config)
        module = model_info.create_module(context=context)
        module_operation = module.operation
        importer = onnx_importer.NodeImporter.define_function(
            model_info.main_graph,
            module_operation,
        )
        importer.import_all()

        if verify:
            module_operation.verify()
        text = module_operation.get_asm(assume_verified=verify)

        return ONNXImportResult(
            text=text,
            initializer_names=initializer_names,
            dynamic_inputs=dynamic_inputs,
            mlir_module=module,
        )


def _sanitize_identifier(name: str) -> str:
    """Return a legal MLIR symbol name derived from ``name``."""
    value = re.sub(r"[^A-Za-z0-9_$.-]", "_", name)
    if not value:
        return "main"
    if value[0].isdigit():
        value = f"_{value}"
    return value


def _onnx_dynamic_inputs(
    graph: onnx.GraphProto,
    initializer_names: tuple[str, ...],
) -> dict[str, tuple[int | str | None, ...]]:
    """Return graph input dimensions after eliding initialized inputs."""
    initializer_set = set(initializer_names)
    result: dict[str, tuple[int | str | None, ...]] = {}
    for graph_input in graph.input:
        if graph_input.name in initializer_set:
            continue
        tensor_type = graph_input.type.tensor_type
        dims: list[int | str | None] = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            elif dim.HasField("dim_param"):
                dims.append(str(dim.dim_param))
            else:
                dims.append(None)
        result[graph_input.name] = tuple(dims)
    return result
