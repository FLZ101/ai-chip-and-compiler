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

import re
from dataclasses import dataclass, field
from typing import Any

from torch_mlir.compiler_utils import (
    BackendLoweringOptions,
    OutputType,
    lower_mlir_module,
    run_pipeline_with_repro_report,
)

from .importer import ONNXImportResult, TorchMLIRImportResult

_TMTENSOR_SCAN_RE = re.compile(
    r"(?P<name>%\w+):2\s*=\s*"
    r"tm_tensor\.scan "
    r"dimension\((?P<dim>\d+)\) "
    r"inclusive\((?P<inclusive>true|false)\) "
    r"ins\((?P<input>%\w+)\s*:\s*"
    r"tensor<(?P<shape>\d+x\d+)x(?P<dtype>[^>]+)>\) "
    r"outs\((?P<output>%\w+),\s*%\w+\s*:\s*"
    r"tensor<\d+x\d+x[^>]+>,\s*tensor<[^>]+>\)\s*"
    r"\{(?P<body>.*?)\}\s*->\s*"
    r"tensor<\d+x\d+x[^>]+>,\s*tensor<[^>]+>",
    re.S,
)


def _lower_tm_tensor_scan_text(text: str) -> str:
    """Lower static rank-2 additive ``tm_tensor.scan`` ops to ``linalg.matmul``.

    torch-mlir's linalg-on-tensors backend may leave ``tm_tensor.scan`` in the
    output for ``torch.aten.cumsum``.  Our ``mgpu-opt`` binary does not link
    the torch-mlir ``tm_tensor`` dialect, so the canonicalizer replaces the
    static inclusive scan along the second dimension with a dense
    lower-triangular matmul.  This is sufficient for GPT-2 causal-mask position
    generation at static sequence lengths.
    """
    lowered = text
    for match in _TMTENSOR_SCAN_RE.finditer(text):
        name = match.group("name")
        dim = int(match.group("dim"))
        inclusive = match.group("inclusive") == "true"
        if dim != 1 or not inclusive:
            raise MLIRCanonicalizationError(
                "only static inclusive tm_tensor.scan along dimension 1 is "
                "currently supported"
            )
        if "arith.addi" not in match.group("body") and "arith.addf" not in match.group(
            "body"
        ):
            raise MLIRCanonicalizationError(
                "tm_tensor.scan replacement currently supports additive scans"
            )
        if name + "#1" in text:
            raise MLIRCanonicalizationError(
                "tm_tensor.scan replacement requires the scan accumulator "
                "result to be unused"
            )

        input_name = match.group("input")
        output_name = match.group("output")
        dtype = match.group("dtype")
        m, n = (int(value) for value in match.group("shape").split("x"))

        rows = []
        for row in range(n):
            values = ["1" if col <= row else "0" for col in range(n)]
            rows.append("[" + ", ".join(values) + "]")
        dense_value = "dense<[" + ", ".join(rows) + "]>"
        dense_type = f"tensor<{n}x{n}x{dtype}>"
        lower_name = "%tm_scan_lower_" + name.lstrip("%")

        line_start = text.rfind("\n", 0, match.start()) + 1
        indent = text[line_start : match.start()]
        replacement = (
            f"{indent}{lower_name} = arith.constant {dense_value} : {dense_type}\n"
            f"{indent}{name} = linalg.matmul "
            f"ins({input_name}, {lower_name} : "
            f"tensor<{m}x{n}x{dtype}>, {dense_type}) "
            f"outs({output_name} : tensor<{m}x{n}x{dtype}>) -> "
            f"tensor<{m}x{n}x{dtype}>"
        )
        lowered = lowered.replace(match.group(0), replacement)
        lowered = lowered.replace(name + "#0", name)
    return lowered


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
        canonical_text = _lower_tm_tensor_scan_text(str(module))

        return CanonicalizedMLIRResult(
            text=canonical_text,
            target=output_type.value,
            source=source,
            mlir_module=module,
        )
