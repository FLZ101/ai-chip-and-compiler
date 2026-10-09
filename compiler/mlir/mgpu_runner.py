"""Invoke ``mgpu-opt`` and extract simulator ISA from its output."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .pipeline import mgpu_opt_pipeline_text

DEFAULT_MODEL_PIPELINE = (
    "builtin.module(one-shot-bufferize,func.func(map-to-mgpu,mgpu-to-isa))"
)


class MGpuOptError(Exception):
    """Raised when ``mgpu-opt`` fails or returns an unexpected module."""


@dataclass(frozen=True)
class MGpuIsaResult:
    """ISA payload extracted from an ``mgpu.isa`` operation."""

    code: str
    grid: tuple[int, ...]
    block: tuple[int, ...]


@dataclass(frozen=True)
class MGpuIsaBundle:
    """One or more ISA programs produced by a full-module pipeline.

    A full real model almost always lowers to many ``mgpu.isa`` operations.
    The programs are kept in program order so a host driver can enqueue them
    into a FIFO launch queue and run them in dependency order.

    For backward compatibility with the earlier single-launch model, the
    bundle also exposes ``code``, ``grid``, and ``block`` properties that
    delegate to the first program.
    """

    programs: tuple[MGpuIsaResult, ...]
    memory_plan: object | None = None

    def __post_init__(self) -> None:
        if not self.programs:
            raise MGpuOptError("model pipeline produced no mgpu.isa operations")

    @property
    def code(self) -> str:
        return self.programs[0].code

    @property
    def grid(self) -> tuple[int, ...]:
        return self.programs[0].grid

    @property
    def block(self) -> tuple[int, ...]:
        return self.programs[0].block


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


def run_model_pipeline(
    mlir_text: str,
    *,
    mgpu_opt: str | Path | None = None,
) -> MGpuIsaBundle:
    """Run the real linalg-on-tensors model pipeline.

    This is the M4.5 entry point.  It buffers tensor-function boundaries,
    maps bufferized linalg compute to ``mgpu.launch``, and emits ISA.  The
    return value now supports full modules, which commonly produce many
    ``mgpu.isa`` operations.
    """
    executable = Path(mgpu_opt or default_mgpu_opt_path())
    if not executable.exists():
        raise MGpuOptError(
            f"mgpu-opt binary not found at {executable}; "
            "build it with `make mgpu-opt` or set MGPU_OPT"
        )

    completed = subprocess.run(
        [str(executable), "-", f"--pass-pipeline={DEFAULT_MODEL_PIPELINE}"],
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

    programs = _extract_isa_results(completed.stdout)
    if not programs:
        raise MGpuOptError("model pipeline produced no mgpu.isa operations")
    return MGpuIsaBundle(programs=tuple(programs))


def run_model_pipeline_with_memory_plan(
    mlir_text: str,
    *,
    mgpu_opt: str | Path | None = None,
    alignment: int = 4096,
) -> MGpuIsaBundle:
    """Run the full-model pipeline with an injected lifetime-aware memory plan.

    This is the Phase C entry point for real models.  It first runs
    one-shot bufferization, collects every static ``memref`` value produced by
    the function, plans non-overlapping byte ranges with the Python allocator
    from :mod:`runtime.memory_plan`, inserts one ``mgpu.memory_layout`` op,
    and then runs the normal ``map-to-mgpu``/``mgpu-to-isa`` passes.

    Subviews, expand-shapes, and collapse-shapes are treated as aliases of
    their source buffer rather than fresh allocations.  This keeps the plan
    representative of real GPU global memory without counting padding views as
    extra storage.
    """
    from runtime.memory_plan import MemoryPlan, PlannedBuffer, plan_memory

    executable = Path(mgpu_opt or default_mgpu_opt_path())
    if not executable.exists():
        raise MGpuOptError(
            f"mgpu-opt binary not found at {executable}; "
            "build it with `make mgpu-opt` or set MGPU_OPT"
        )

    bufferized_text = _run_mgpu_opt_text(
        mlir_text,
        "builtin.module(one-shot-bufferize)",
        executable,
    )
    bufferized_values = _collect_bufferized_values(bufferized_text)
    if not bufferized_values:
        raise MGpuOptError("bufferized module contains no static memref buffers")

    roots = [value for value in bufferized_values if value.parent is None]
    aliases = [value for value in bufferized_values if value.parent is not None]

    planned_buffers = [
        PlannedBuffer(
            name=value.ssa_name,
            numel=value.numel,
            dtype=value.dtype,
            live_from=value.live_from,
            live_until=value.live_until,
        )
        for value in roots
    ]
    memory_plan = plan_memory(planned_buffers, alignment=alignment)

    offset_by_value = {
        value.ssa_name: memory_plan.offset(value.ssa_name) for value in roots
    }
    for value in aliases:
        if value.parent not in offset_by_value:
            raise MGpuOptError(
                f"alias {value.ssa_name} refers to unknown parent {value.parent}"
            )
        offset_by_value[value.ssa_name] = offset_by_value[value.parent]

    injected_text = _inject_memory_layout(
        bufferized_text,
        bufferized_values,
        offset_by_value,
    )
    programs = _run_mgpu_opt_bundle(
        injected_text,
        "builtin.module(func.func(map-to-mgpu,mgpu-to-isa))",
        executable,
    )
    return MGpuIsaBundle(programs=tuple(programs), memory_plan=memory_plan)


def enqueue_isa_bundle(driver: object, bundle: MGpuIsaBundle) -> None:
    """Enqueue all programs from *bundle* on a runtime :class:`Driver`.

    The programs are enqueued without immediately flushing so callers can
    enqueue additional work and then call ``driver.synchronize()`` once.  The
    runtime executes its queue in FIFO order, which is the dependency order
    produced by ``mgpu-opt``.
    """
    for program in bundle.programs:
        driver.enqueue(  # type: ignore[attr-defined]
            program.code,
            grid_dim=program.grid,
            block_dim=program.block,
        )


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


def _extract_isa_results(mlir_text: str) -> list[MGpuIsaResult]:
    return [
        MGpuIsaResult(
            code=_unescape_mlir_string(attrs["code"]),
            grid=_parse_dim_string(_unescape_mlir_string(attrs["grid"])),
            block=_parse_dim_string(_unescape_mlir_string(attrs["block"])),
        )
        for attrs in _extract_isa_attrs(mlir_text)
    ]


def _run_mgpu_opt_text(
    mlir_text: str,
    pipeline: str,
    executable: Path,
) -> str:
    completed = subprocess.run(
        [str(executable), "-", f"--pass-pipeline={pipeline}"],
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
    return completed.stdout


def _run_mgpu_opt_bundle(
    mlir_text: str,
    pipeline: str,
    executable: Path,
) -> list[MGpuIsaResult]:
    output = _run_mgpu_opt_text(mlir_text, pipeline, executable)
    programs = _extract_isa_results(output)
    if not programs:
        raise MGpuOptError("model pipeline produced no mgpu.isa operations")
    return programs


class _BufferizedValue:
    """One static memref SSA value parsed from a bufferized MLIR module."""

    def __init__(
        self,
        *,
        ssa_name: str,
        mlir_type: str,
        numel: int,
        dtype: str,
        live_from: int,
        live_until: int | None,
        parent: str | None,
    ):
        self.ssa_name = ssa_name
        self.mlir_type = mlir_type
        self.numel = numel
        self.dtype = dtype
        self.live_from = live_from
        self.live_until = live_until
        self.parent = parent


_ROOT_MEMREF_OPS = (
    "bufferization.to_buffer",
    "memref.get_global",
    "memref.alloc",
)
_ALIAS_MEMREF_OPS = (
    "memref.subview",
    "memref.collapse_shape",
    "memref.expand_shape",
)

_DTYPE_MAP = {
    "f32": "f32",
    "f64": "f64",
    "f16": "f16",
    "bf16": "bf16",
    "i32": "i32",
    "i64": "i64",
    "i1": "i1",
    "i8": "i8",
    "ui8": "u8",
    "i4": "i4",
    "ui4": "u4",
    "f8E4M3FN": "f8_e4m3",
    "f8E5M2": "f8_e5m2",
}


def _collect_bufferized_values(mlir_text: str) -> list[_BufferizedValue]:
    pattern = re.compile(
        r"^\s*(%[A-Za-z0-9_]+)\s*=\s*"
        r"(bufferization\.to_buffer|memref\.get_global|memref\.alloc|"
        r"memref\.subview|memref\.collapse_shape|memref\.expand_shape)\b",
        re.MULTILINE,
    )
    last_use_by_ssa = _build_last_use_map(mlir_text)

    values: list[_BufferizedValue] = []
    for match in pattern.finditer(mlir_text):
        line_end = mlir_text.find("\n", match.start())
        if line_end == -1:
            line_end = len(mlir_text)
        line = mlir_text[match.start() : line_end]

        ssa_name = match.group(1)
        op_name = match.group(2)
        if op_name in (
            "bufferization.to_buffer",
            "memref.subview",
            "memref.collapse_shape",
            "memref.expand_shape",
        ):
            memref_type = _extract_result_memref_type(line)
        else:
            memref_type = _extract_balanced_memref_type(line)
        if memref_type is None:
            continue
        parsed = _parse_memref_type(memref_type)
        if parsed is None:
            continue
        shape, mlir_dtype = parsed
        dtype = _DTYPE_MAP.get(mlir_dtype)
        if dtype is None:
            continue

        numel = 1
        for dim in shape:
            numel *= dim
        if numel <= 0:
            continue

        line_number = mlir_text.count("\n", 0, match.start()) + 1
        parent = None
        if op_name in _ALIAS_MEMREF_OPS:
            parent = _parse_parent_ssa(line, op_name)
            if parent is None:
                continue

        live_until = None
        if op_name in _ALIAS_MEMREF_OPS:
            live_until = last_use_by_ssa.get(ssa_name, line_number)
        elif op_name != "memref.get_global" and op_name != "bufferization.to_buffer":
            live_until = last_use_by_ssa.get(ssa_name, line_number)

        values.append(
            _BufferizedValue(
                ssa_name=ssa_name,
                mlir_type=memref_type,
                numel=numel,
                dtype=dtype,
                live_from=line_number,
                live_until=live_until,
                parent=parent,
            )
        )

    return values


def _extract_balanced_memref_type(line: str) -> str | None:
    start = line.find("memref<")
    if start == -1:
        return None
    index = start + len("memref<")
    depth = 1
    while index < len(line) and depth:
        if line[index] == "<":
            depth += 1
        elif line[index] == ">":
            depth -= 1
        index += 1
    if depth:
        return None
    return line[start:index]


def _extract_result_memref_type(line: str) -> str | None:
    for marker in (" to ", " into "):
        marker_index = line.rfind(marker)
        if marker_index != -1:
            return _extract_balanced_memref_type(line[marker_index + len(marker) :])
    return _extract_balanced_memref_type(line)


def _parse_memref_type(memref_type: str) -> tuple[tuple[int, ...], str] | None:
    if not memref_type.startswith("memref<") or not memref_type.endswith(">"):
        return None
    inner = memref_type[len("memref<") : -1]
    shape_and_dtype = _split_top_level(inner, ",")[0]
    parts = shape_and_dtype.split("x")
    if len(parts) == 1:
        dtype = parts[0]
        shape: tuple[int, ...] = ()
    elif len(parts) >= 2:
        dtype = parts[-1]
        try:
            shape = tuple(int(part) for part in parts[:-1])
        except ValueError:
            return None
    else:
        return None
    return shape, dtype


def _split_top_level(value: str, separator: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif char == separator and depth == 0:
            parts.append(value[start:index])
            start = index + 1
    parts.append(value[start:])
    return parts


def _parse_parent_ssa(line: str, op_name: str) -> str | None:
    if op_name == "memref.subview":
        match = re.search(r"memref\.subview\s+(%[A-Za-z0-9_]+)", line)
    else:
        match = re.search(
            r"memref\.(?:collapse_shape|expand_shape)\s+(%[A-Za-z0-9_]+)", line
        )
    if not match:
        return None
    return match.group(1)


def _build_last_use_map(mlir_text: str) -> dict[str, int]:
    last_use: dict[str, int] = {}
    for line_number, line in enumerate(mlir_text.splitlines(), 1):
        for match in re.finditer(r"%[A-Za-z0-9_]+", line):
            last_use[match.group(0)] = line_number
    return last_use


def _inject_memory_layout(
    bufferized_text: str,
    bufferized_values: list[_BufferizedValue],
    offset_by_value: dict[str, int],
) -> str:
    names = [value.ssa_name for value in bufferized_values]
    types = [value.mlir_type for value in bufferized_values]
    offsets = [offset_by_value[value.ssa_name] for value in bufferized_values]
    layout_line = (
        "    mgpu.memory_layout "
        + ", ".join(names)
        + " offsets = ["
        + ", ".join(str(offset) for offset in offsets)
        + "] : "
        + ", ".join(types)
    )

    func_start = bufferized_text.find("func.func @")
    if func_start == -1:
        raise MGpuOptError("bufferized module has no func.func")
    func_body = bufferized_text[func_start:]
    return_match = re.search(r"\n\s+return\b", func_body)
    if return_match:
        insert_pos = func_start + return_match.start()
    else:
        closing_brace = func_body.rfind("}")
        if closing_brace == -1:
            raise MGpuOptError("bufferized func.func has no body")
        insert_pos = func_start + closing_brace
    return (
        bufferized_text[:insert_pos] + layout_line + "\n" + bufferized_text[insert_pos:]
    )


def _unescape_mlir_string(value: str) -> str:
    return value.replace("\\0A", "\n").replace("\\22", '"').replace("\\5C", "\\")


def _parse_dim_string(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split("x"))
    except ValueError as exc:
        raise MGpuOptError(f"invalid launch dimension string: {value!r}") from exc
