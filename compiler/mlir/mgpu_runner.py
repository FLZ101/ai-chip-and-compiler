"""Invoke ``mgpu-opt`` and extract simulator ISA from its output."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

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

    isa_attrs = _extract_isa_attrs(completed.stdout)
    if len(isa_attrs) != 1:
        raise MGpuOptError(
            f"expected exactly one mgpu.isa operation, found {len(isa_attrs)}"
        )

    attrs = isa_attrs[0]
    code = _unescape_mlir_string(attrs["code"])
    grid = _parse_dim_string(_unescape_mlir_string(attrs["grid"]))
    block = _parse_dim_string(_unescape_mlir_string(attrs["block"]))
    return MGpuIsaResult(code=code, grid=grid, block=block)


def lower_with_mgpu_opt(
    graph: object,
    mlir_text: str,
    layout: dict[str, object],
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Adapter matching the E2E ``LoweringFn`` contract."""
    from .mlir_builder import MLIRBuilder

    builder = MLIRBuilder()
    input_names = [value.name for value in graph.inputs]
    output_names = [value.name for value in graph.outputs]
    buffers = [f"%arg{index}" for index in range(len(input_names))]
    buffers.extend(f"%v{index}" for index in range(len(output_names)))
    types = [builder._effective_input_type(graph, value) for value in graph.inputs]
    types.extend(
        builder._memref_type_for(value.shape, value.dtype) for value in graph.outputs
    )
    offsets = [layout[name].base_addr for name in input_names + output_names]

    layout_line = (
        "    mgpu.memory_layout "
        + ", ".join(buffers)
        + " offsets = ["
        + ", ".join(str(offset) for offset in offsets)
        + "] : "
        + ", ".join(types)
    )
    injected = mlir_text.replace(
        "    mgpu.launch",
        layout_line + "\n    mgpu.launch",
        1,
    )

    result = run_mgpu_pipeline(injected)
    return result.code, (result.grid, result.block)


def _extract_isa_attrs(mlir_text: str) -> list[dict[str, str]]:
    operations = re.findall(r'"mgpu\.isa"\(\)\s*<\{([^}]*)\}>', mlir_text)
    results: list[dict[str, str]] = []
    for body in operations:
        attrs: dict[str, str] = {}
        for name, value in re.findall(
            r'\b(code|grid|block)\s*=\s*"((?:\\.|[^"])*)"', body
        ):
            attrs[name] = value
        if set(attrs) == {"code", "grid", "block"}:
            results.append(attrs)
    return results


def _unescape_mlir_string(value: str) -> str:
    return value.replace("\\0A", "\n").replace("\\22", '"').replace("\\5C", "\\")


def _parse_dim_string(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split("x"))
    except ValueError as exc:
        raise MGpuOptError(f"invalid launch dimension string: {value!r}") from exc
