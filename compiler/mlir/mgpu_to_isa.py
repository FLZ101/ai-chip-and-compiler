"""Reference ``mgpu-to-isa`` lowering for the simulator target.

Phase 3.6 uses the existing deterministic Model IR code generators as the
reference ISA backend.  The generic MLIR-to-ISA pass will later consume the
``mgpu`` dialect produced by ``mgpu-opt``; for now this module provides the
same lowering entry point so end-to-end tests can validate simulator output.
"""

from __future__ import annotations

from typing import Any

from ..codegen import lower_kernel
from ..model_ir import Graph


class MGpuToIsaError(Exception):
    """Raised when mgpu-to-isa cannot lower a graph."""


def lower_mgpu_to_isa(
    graph: Graph,
    mlir_text: str,
    layout: dict[str, Any],
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Lower a validated Model IR graph and MLIR skeleton to simulator ISA."""
    if "mgpu.launch" not in mlir_text:
        raise MGpuToIsaError("MLIR text does not contain an mgpu.launch")
    return lower_kernel(graph, mlir_text, layout)
