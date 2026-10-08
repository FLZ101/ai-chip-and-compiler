"""Invoke ``mgpu-opt`` and extract simulator ISA from its output."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from torch_mlir import ir

from .pipeline import mgpu_opt_pipeline_text


class MGpuOptError(Exception):
    """Raised when ``mgpu-opt`` fails or returns an unexpected module."""


@dataclass(frozen=True)
class MGpuIsaResult:
    """ISA payload extracted from an ``mgpu.isa`` operation."""

    code: str
    grid: tuple[int, ...]
    block: tuple[int, ...]


def default_mgpu_opt_path() -> Path:
    """Return the path to the out-of-tree ``mgpu-opt`` binary."""
    override = os.environ.get("MGPU_OPT")
    if override:
        return Path(override)
    return (
        Path(__file__).resolve().parents[2]
        / ".deps"
        / "mgpu-build"
        / "bin"
        / "mgpu-opt"
    )


def run_mgpu_pipeline(
    mlir_text: str,
    *,
    mgpu_opt: str | Path | None = None,
    pipeline: str | None = None,
) -> MGpuIsaResult:
    """Run ``mgpu-opt`` on *mlir_text* and extract one ``mgpu.isa`` op."""
    executable = Path(mgpu_opt or default_mgpu_opt_path())
    if not executable.exists():
        raise MGpuOptError(
            f"mgpu-opt binary not found at {executable}; "
            "build it with `make mgpu-opt` or set MGPU_OPT"
        )

    pass_pipeline = pipeline or mgpu_opt_pipeline_text()
    completed = subprocess.run(
        [str(executable), "-", f"--pass-pipeline={pass_pipeline}"],
        input=mlir_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise MGpuOptError(
            f"mgpu-opt failed with exit code {completed.returncode}:\n"
            f"{completed.stderr}"
        )

    module = _parse_module(completed.stdout)
    isa_ops = [op for op in _walk_module_operations(module) if op.name == "mgpu.isa"]
    if len(isa_ops) != 1:
        raise MGpuOptError(
            f"expected exactly one mgpu.isa operation, found {len(isa_ops)}"
        )

    op = isa_ops[0]
    code = _string_attr(op, "code")
    grid = _dim_tuple_attr(op, "grid")
    block = _dim_tuple_attr(op, "block")
    return MGpuIsaResult(code=code, grid=grid, block=block)


def lower_with_mgpu_opt(
    graph: object,
    mlir_text: str,
    layout: dict[str, object],
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Adapter matching the E2E ``LoweringFn`` contract."""
    del graph, layout  # Phase 0 does not yet need them; later phases will.
    result = run_mgpu_pipeline(mlir_text)
    return result.code, (result.grid, result.block)


def _parse_module(mlir_text: str):
    with ir.Context() as context:
        context.allow_unregistered_dialects = True
        return ir.Module.parse(mlir_text)


def _walk_operations(op):
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield child
                yield from _walk_operations(child)


def _walk_module_operations(module):
    for child in module.body.operations:
        yield child
        yield from _walk_operations(child)


def _string_attr(op, name: str) -> str:
    value = op.attributes[name].value
    if not isinstance(value, str):
        raise MGpuOptError(f"attribute {name!r} on {op.name} is not a string")
    return value


def _dim_tuple_attr(op, name: str) -> tuple[int, ...]:
    value = _string_attr(op, name)
    try:
        return tuple(int(part) for part in value.split("x"))
    except ValueError as exc:
        raise MGpuOptError(
            f"attribute {name!r} on {op.name} is not an x-separated tuple: {value!r}"
        ) from exc
