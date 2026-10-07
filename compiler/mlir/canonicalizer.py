"""Canonicalize imported torch-mlir modules.

Phase 2.3 runs torch-mlir's decomposition, shape inference, canonicalization,
and backend lowering pipelines on the raw ``torch`` dialect modules produced by
the Phase 2.1 FX importer and the Phase 2.2 ONNX importer.

The result is a deterministic high-level module in one of two common backend
dialects:

* ``linalg-on-tensors`` (default): ``linalg``, ``tensor``, ``arith``, and
  ``math``.
* ``tosa``: ``tosa`` plus the supporting ``tensor``/``arith``/``math`` ops.

``memref`` intentionally arrives later, in Phase 3.3 bufferization, after tiling
and mapping decisions have been made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch_mlir.compiler_utils import (
    BackendLoweringOptions,
    OutputType,
    lower_mlir_module,
    run_pipeline_with_repro_report,
)

from .importer import ONNXImportResult, TorchMLIRImportResult


class MLIRCanonicalizationError(Exception):
    """Raised when an imported module cannot be canonicalized."""


@dataclass(frozen=True)
class CanonicalizedMLIRResult:
    """A torch-mlir module after canonicalization and backend lowering.

    Attributes:
        text: Stable MLIR text after lowering.
        target: The selected backend dialect family, either
            ``linalg-on-tensors`` or ``tosa``.
        source: The original import source, either ``fx`` or ``onnx``.
        mlir_module: The lowered in-memory ``torch_mlir.ir.Module``.
    """

    text: str
    target: str
    source: str
    mlir_module: Any = field(repr=False, compare=False)


class MLIRCanonicalizer:
    """Canonicalize a raw import result into a common high-level dialect."""

    def canonicalize(
        self,
        imported: TorchMLIRImportResult | ONNXImportResult,
        *,
        target: str = "linalg-on-tensors",
        allow_non_finites: bool = True,
    ) -> CanonicalizedMLIRResult:
        """Run the correct torch-mlir pre-backend and backend pipelines.

        Raw FX and raw ONNX modules need different preparation passes before the
        shared backend lowering pipeline.  Both inputs are then lowered to the
        requested ``OutputType``.
        """
        output_type = OutputType.get(target)
        if output_type not in {OutputType.LINALG_ON_TENSORS, OutputType.TOSA}:
            raise MLIRCanonicalizationError(
                "Phase 2.3 canonicalization supports linalg-on-tensors or tosa"
            )

        if isinstance(imported, TorchMLIRImportResult):
            source = "fx"
            preparation_pipeline = (
                "builtin.module("
                "func.func(torch-match-quantized-custom-ops), "
                "torchdynamo-export-to-torch-backend-pipeline"
                ")"
            )
            description = "Lowering FX torch IR to torch backend IR"
        elif isinstance(imported, ONNXImportResult):
            source = "onnx"
            preparation_pipeline = (
                "builtin.module(torch-onnx-to-torch-backend-pipeline)"
            )
            description = "Lowering ONNX torch IR to torch backend IR"
        else:
            raise MLIRCanonicalizationError(
                f"unsupported import result type: {type(imported).__name__}"
            )

        module = imported.mlir_module
        run_pipeline_with_repro_report(
            module,
            preparation_pipeline,
            description,
        )
        lower_mlir_module(
            False,
            output_type,
            module,
            BackendLoweringOptions(allow_non_finites=allow_non_finites),
        )
        module.operation.verify()

        return CanonicalizedMLIRResult(
            text=str(module),
            target=output_type.value,
            source=source,
            mlir_module=module,
        )
