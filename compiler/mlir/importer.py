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

import torch
from torch_mlir import fx as torch_mlir_fx
from torch_mlir.compiler_utils import OutputType


class TorchMLIRImportError(Exception):
    """Raised when a PyTorch program cannot be imported into MLIR."""


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


def _sanitize_identifier(name: str) -> str:
    """Return a legal MLIR symbol name derived from ``name``."""
    value = re.sub(r"[^A-Za-z0-9_$.-]", "_", name)
    if not value:
        return "main"
    if value[0].isdigit():
        value = f"_{value}"
    return value
