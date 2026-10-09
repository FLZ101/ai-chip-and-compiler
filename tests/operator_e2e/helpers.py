"""Shared end-to-end operator test harness.

Step 007 establishes one pipeline for every operator test:

```text
model graph -> MLIR -> ISA -> simulator -> device buffers
```

The MLIR-to-ISA stage is intentionally a pluggable callable so future kernel
implementations can be added without changing the harness.  A small vector-add
lowering is included here as the first reference implementation.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from compiler.mlir import MLIRBuilder
from compiler.model_ir import Graph, Op
from hw.device import Device, create_device, launch
from hw.isa import parse_asm
from hw.timing import ProfileCounters


@dataclass(frozen=True)
class TensorBuffer:
    """A named tensor placed in simulated global memory."""

    name: str
    base_addr: int
    shape: tuple[int, ...]
    dtype: str = "f32"

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


MemoryLayout = dict[str, TensorBuffer]
LoweringFn = Callable[
    [Graph, str, MemoryLayout],
    tuple[str, tuple[tuple[int, ...], tuple[int, ...]]],
]
ReferenceFn = Callable[[dict[str, np.ndarray]], list[np.ndarray]]


@dataclass
class OperatorE2EResult:
    """Outputs and artifacts from one end-to-end operator run."""

    outputs: list[np.ndarray]
    mlir_text: str
    isa_text: str
    cycles: int
    profile: ProfileCounters | None = None


def allocate_global_buffers(
    device: Device,
    tensors: dict[str, np.ndarray],
    *,
    alignment: int = 4,
) -> MemoryLayout:
    """Write host tensors into global memory and return their placements."""
    layout: MemoryLayout = {}
    cursor = 0

    for name, array in tensors.items():
        if cursor > 0:
            cursor = ((cursor + alignment - 1) // alignment) * alignment

        data = np.asarray(array, dtype=np.float32)
        base_addr = cursor
        if base_addr + data.nbytes > len(device.global_memory.data):
            raise ValueError(f"global memory exhausted while placing tensor {name!r}")
        device.global_memory.load_floats(base_addr, data.reshape(-1).tolist())
        layout[name] = TensorBuffer(
            name=name,
            base_addr=base_addr,
            shape=tuple(data.shape),
            dtype="f32",
        )
        cursor = base_addr + data.nbytes

    return layout


def read_tensor(device: Device, buffer: TensorBuffer) -> np.ndarray:
    """Read a placed tensor back from simulated global memory."""
    values = device.global_memory.read_floats(buffer.base_addr, buffer.numel)
    return np.asarray(values, dtype=np.float32).reshape(buffer.shape)


def compare_against_numpy(
    actuals: list[np.ndarray],
    expected: list[np.ndarray],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Compare simulator outputs with NumPy golden tensors."""
    if len(actuals) != len(expected):
        raise AssertionError(f"expected {len(expected)} outputs, got {len(actuals)}")
    for index, (actual, golden) in enumerate(zip(actuals, expected)):
        np.testing.assert_allclose(
            actual,
            np.asarray(golden, dtype=np.float32),
            rtol=rtol,
            atol=atol,
            err_msg=f"output {index} differs from NumPy reference",
        )


def run_operator_e2e(
    device: Device,
    graph: Graph,
    inputs: dict[str, np.ndarray],
    expected: list[np.ndarray] | ReferenceFn | None = None,
    *,
    lowering: LoweringFn,
    profile: bool = False,
    progress: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> OperatorE2EResult:
    """Run one operator through the shared compiler + simulator pipeline."""
    graph.validate()
    mlir_text = MLIRBuilder().build(graph)

    input_names = {value.name for value in graph.inputs}
    if set(inputs) != input_names:
        raise ValueError(
            f"inputs must match graph inputs: expected {sorted(input_names)}, "
            f"got {sorted(inputs)}"
        )

    for value in graph.inputs:
        array = np.asarray(inputs[value.name])
        if tuple(array.shape) != value.shape:
            raise ValueError(
                f"input {value.name!r} has shape {array.shape}, "
                f"expected {value.shape}"
            )

    intermediate_tensors = {
        value.name: np.zeros(value.shape, dtype=np.float32)
        for node in graph.nodes
        for value in node.outputs
    }
    layout = allocate_global_buffers(device, {**inputs, **intermediate_tensors})

    if len(graph.nodes) == 1:
        if progress:
            node = graph.nodes[0]
            print(
                f"[model-e2e] op 1/1: {node.op.value} -> "
                f"{[value.name for value in node.outputs]}",
                file=sys.stderr,
                flush=True,
            )
        isa_text, (grid_dim, block_dim) = lowering(graph, mlir_text, layout)
        program = parse_asm(isa_text)
        profile_snapshot = launch(
            device,
            program,
            grid_dim=grid_dim,
            block_dim=block_dim,
            profile=profile,
        )
    else:
        isa_parts: list[str] = []
        profile_snapshot = None
        for index, node in enumerate(graph.nodes, 1):
            if progress:
                print(
                    f"[model-e2e] op {index}/{len(graph.nodes)}: "
                    f"{node.op.value} -> {[value.name for value in node.outputs]}",
                    file=sys.stderr,
                    flush=True,
                )
            subgraph = Graph(
                name=node.name,
                inputs=list(node.inputs),
                outputs=list(node.outputs),
                nodes=[node],
            )
            subgraph.validate()
            sub_mlir_text = MLIRBuilder().build(subgraph)
            node_isa, (grid_dim, block_dim) = lowering(subgraph, sub_mlir_text, layout)
            isa_parts.append(node_isa)
            program = parse_asm(node_isa)
            profile_snapshot = launch(
                device,
                program,
                grid_dim=grid_dim,
                block_dim=block_dim,
                profile=profile,
            )
        isa_text = "\n\n".join(isa_parts)

    outputs = [read_tensor(device, layout[value.name]) for value in graph.outputs]
    if expected is not None:
        golden = expected(inputs) if callable(expected) else expected
        compare_against_numpy(outputs, golden, rtol=rtol, atol=atol)

    return OperatorE2EResult(
        outputs=outputs,
        mlir_text=mlir_text,
        isa_text=isa_text,
        cycles=device.clock,
        profile=profile_snapshot,
    )


def vector_add_lowering(
    graph: Graph,
    mlir_text: str,
    layout: MemoryLayout,
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Lower a single rank-1 elementwise add graph to the simulator ISA."""
    del mlir_text  # Reserved for the generic MLIR-to-ISA pass in later steps.

    if len(graph.nodes) != 1 or graph.nodes[0].op != Op.ADD:
        raise NotImplementedError(
            "vector_add_lowering supports exactly one elementwise add node"
        )

    node = graph.nodes[0]
    left_name, right_name = (value.name for value in node.inputs)
    output_name = graph.outputs[0].name
    left = layout[left_name]
    right = layout[right_name]
    output = layout[output_name]

    if left.shape != right.shape or left.shape != output.shape:
        raise NotImplementedError("vector_add_lowering requires equal tensor shapes")
    if len(left.shape) != 1:
        raise NotImplementedError("vector_add_lowering supports rank-1 vectors")

    n = left.shape[0]
    isa_text = f"""
MOV R4, {left.base_addr}
MUL R0, tid.x, 4
ADD R0, R0, R4
LDG R1, [R0]
LDG R2, [R0 + {right.base_addr - left.base_addr}]
ADD R3, R1, R2
STG [R0 + {output.base_addr - left.base_addr}], R3
EXIT
"""
    return isa_text, ((1,), (n,))


def make_device(global_size: int = 1 << 20, sm_count: int = 4) -> Device:
    """Create the default device used by operator E2E tests."""
    return create_device(global_size=global_size, sm_count=sm_count)


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _software_request(
    node: object,
    layout: MemoryLayout,
) -> dict[str, object]:
    op = node.op
    values = node.inputs
    output = node.outputs[0]

    def offset(value: object) -> int:
        return layout[value.name].base_addr

    if op == Op.CONV2D:
        strides = _pair(node.attrs.get("strides", 1))
        padding = _pair(node.attrs.get("padding", 0))
        dilation = _pair(node.attrs.get("dilation", 1))
        has_bias = len(values) == 3
        return {
            "op": "conv2d",
            "input_shape": values[0].shape,
            "weight_shape": values[1].shape,
            "output_shape": output.shape,
            "strides": strides,
            "paddings": padding,
            "dilations": dilation,
            "has_bias": has_bias,
            "input_offset": offset(values[0]),
            "weight_offset": offset(values[1]),
            "bias_offset": offset(values[2]) if has_bias else 0,
            "output_offset": offset(output),
        }
    if op == Op.MATMUL:
        return {
            "op": "matmul",
            "input_shape": values[0].shape,
            "weight_shape": values[1].shape,
            "output_shape": output.shape,
            "input_offset": offset(values[0]),
            "weight_offset": offset(values[1]),
            "bias_offset": 0,
            "output_offset": offset(output),
        }
    if op == Op.LINEAR:
        has_bias = len(values) == 3
        return {
            "op": "linear",
            "input_shape": values[0].shape,
            "weight_shape": values[1].shape,
            "output_shape": output.shape,
            "has_bias": has_bias,
            "input_offset": offset(values[0]),
            "weight_offset": offset(values[1]),
            "bias_offset": offset(values[2]) if has_bias else 0,
            "output_offset": offset(output),
        }
    if op == Op.RELU:
        return {
            "op": "relu",
            "input_shape": values[0].shape,
            "output_shape": output.shape,
            "input_offset": offset(values[0]),
            "output_offset": offset(output),
        }
    if op == Op.ADD:
        return {
            "op": "add",
            "input_shape": values[0].shape,
            "output_shape": output.shape,
            "input_offset": offset(values[0]),
            "input2_offset": offset(values[1]),
            "output_offset": offset(output),
        }
    if op == Op.BATCH_NORM:
        return {
            "op": "batch_norm",
            "input_shape": values[0].shape,
            "weight_shape": values[1].shape,
            "output_shape": output.shape,
            "epsilon": float(node.attrs.get("epsilon", 1e-5)),
            "input_offset": offset(values[0]),
            "weight_offset": offset(values[1]),
            "bias_offset": offset(values[2]),
            "mean_offset": offset(values[3]),
            "var_offset": offset(values[4]),
            "output_offset": offset(output),
        }
    if op == Op.GLOBAL_AVG_POOL:
        return {
            "op": "global_avg_pool",
            "input_shape": values[0].shape,
            "output_shape": output.shape,
            "input_offset": offset(values[0]),
            "output_offset": offset(output),
        }
    if op in {Op.MAX_POOL2D, Op.AVG_POOL2D}:
        kernel = _pair(node.attrs.get("kernel", 2))
        strides = _pair(node.attrs.get("strides", kernel))
        padding = _pair(node.attrs.get("padding", 0))
        return {
            "op": "max_pool2d" if op == Op.MAX_POOL2D else "avg_pool2d",
            "input_shape": values[0].shape,
            "output_shape": output.shape,
            "strides": strides,
            "paddings": padding,
            "dilations": (1, 1),
            "kernel": kernel,
            "input_offset": offset(values[0]),
            "output_offset": offset(output),
        }
    if op == Op.PAD:
        pads = tuple(node.attrs.get("pads", ((0, 0),)))
        before = int(pads[2][0]) if len(pads) > 2 else int(pads[0][0])
        after = int(pads[2][1]) if len(pads) > 2 else int(pads[0][1])
        return {
            "op": "pad",
            "input_shape": values[0].shape,
            "output_shape": output.shape,
            "paddings": (before, after),
            "pad_value": float(node.attrs.get("value", 0.0)),
            "input_offset": offset(values[0]),
            "output_offset": offset(output),
        }
    if op == Op.RESHAPE:
        return {
            "op": "reshape",
            "input_shape": values[0].shape,
            "output_shape": output.shape,
            "input_offset": offset(values[0]),
            "output_offset": offset(output),
        }
    if op == Op.EMBEDDING_GATHER:
        return {
            "op": "embedding_gather",
            "input_shape": values[0].shape,
            "weight_shape": values[1].shape,
            "output_shape": output.shape,
            "input_offset": offset(values[0]),
            "weight_offset": offset(values[1]),
            "output_offset": offset(output),
        }
    if op == Op.ARGMAX:
        return {
            "op": "argmax",
            "input_shape": values[0].shape,
            "output_shape": output.shape,
            "input_offset": offset(values[0]),
            "output_offset": offset(output),
            "axis": int(node.attrs.get("axis", -1)),
        }
    raise NotImplementedError(f"no software kernel for {op.value!r}")


def run_software_model_e2e(
    device: Device,
    graph: Graph,
    inputs: dict[str, np.ndarray],
    expected: list[np.ndarray] | ReferenceFn | None = None,
    *,
    progress: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> OperatorE2EResult:
    cpp_sim_dir = Path(__file__).resolve().parents[2] / ".deps" / "cpp-sim-build"
    if str(cpp_sim_dir) not in sys.path:
        sys.path.insert(0, str(cpp_sim_dir))
    import cpp_sim

    graph.validate()
    if set(inputs) != {value.name for value in graph.inputs}:
        raise ValueError("inputs must match graph inputs")
    for value in graph.inputs:
        array = np.asarray(inputs[value.name])
        if tuple(array.shape) != value.shape:
            raise ValueError(
                f"input {value.name!r} has shape {array.shape}, expected {value.shape}"
            )

    intermediate_tensors = {
        value.name: np.zeros(value.shape, dtype=np.float32)
        for node in graph.nodes
        for value in node.outputs
    }
    layout = allocate_global_buffers(device, {**inputs, **intermediate_tensors})

    for index, node in enumerate(graph.nodes, 1):
        if progress:
            print(
                f"[model-e2e] software op {index}/{len(graph.nodes)}: "
                f"{node.op.value} -> {[value.name for value in node.outputs]}",
                file=sys.stderr,
                flush=True,
            )
        try:
            request = _software_request(node, layout)
        except NotImplementedError:
            _run_python_software_node(device, node, layout)
            continue
        if not cpp_sim.software_run(device.global_memory.data, request):
            _run_python_software_node(device, node, layout)

    outputs = [read_tensor(device, layout[value.name]) for value in graph.outputs]
    if expected is not None:
        golden = expected(inputs) if callable(expected) else expected
        compare_against_numpy(outputs, golden, rtol=rtol, atol=atol)
    return OperatorE2EResult(
        outputs=outputs,
        mlir_text="software model runner",
        isa_text="",
        cycles=0,
        profile=None,
    )


def _read_node_input(device: Device, value: object, layout: MemoryLayout) -> np.ndarray:
    buffer = layout[value.name]
    return read_tensor(device, buffer)


def _write_node_output(
    device: Device,
    value: object,
    layout: MemoryLayout,
    array: np.ndarray,
) -> None:
    device.global_memory.load_floats(
        layout[value.name].base_addr,
        np.asarray(array, dtype=np.float32).reshape(-1).tolist(),
    )


def _run_python_software_node(
    device: Device,
    node: object,
    layout: MemoryLayout,
) -> None:
    op = node.op
    output = node.outputs[0]

    if op == Op.TRANSPOSE:
        value = _read_node_input(device, node.inputs[0], layout)
        perm = node.attrs.get("perm", tuple(reversed(range(value.ndim))))
        _write_node_output(device, output, layout, np.transpose(value, perm))
        return

    if op == Op.CONCAT:
        values = [_read_node_input(device, value, layout) for value in node.inputs]
        axis = int(node.attrs.get("axis", 0))
        _write_node_output(device, output, layout, np.concatenate(values, axis=axis))
        return

    if op == Op.SLICE:
        value = _read_node_input(device, node.inputs[0], layout)
        starts = tuple(int(x) for x in node.attrs.get("starts", (0,) * value.ndim))
        ends = tuple(int(x) for x in node.attrs.get("ends", value.shape))
        axes = tuple(int(x) for x in node.attrs.get("axes", tuple(range(value.ndim))))
        steps = tuple(int(x) for x in node.attrs.get("steps", (1,) * value.ndim))
        slices = [slice(None)] * value.ndim
        for start, end, axis, step in zip(starts, ends, axes, steps):
            slices[axis] = slice(start, end, step)
        _write_node_output(device, output, layout, value[tuple(slices)])
        return

    if op == Op.RESIZE:
        value = _read_node_input(device, node.inputs[0], layout)
        sizes = node.attrs.get("sizes")
        if sizes is None:
            scales = tuple(float(x) for x in node.attrs.get("scales", ()))
            sizes = tuple(
                max(int(dim * scale), 1) for dim, scale in zip(value.shape, scales)
            )
        # Nearest-neighbor resize.
        output_array = np.empty(tuple(int(x) for x in sizes), dtype=np.float32)
        for out_index in np.ndindex(output_array.shape):
            in_index = tuple(
                min(
                    int(out_index[axis] * value.shape[axis] / output_array.shape[axis]),
                    value.shape[axis] - 1,
                )
                for axis in range(value.ndim)
            )
            output_array[out_index] = value[in_index]
        _write_node_output(device, output, layout, output_array)
        return

    if op == Op.SOFTMAX:
        value = _read_node_input(device, node.inputs[0], layout)
        axis = int(node.attrs.get("axis", -1))
        shifted = value - np.max(value, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        _write_node_output(
            device, output, layout, exp / np.sum(exp, axis=axis, keepdims=True)
        )
        return

    if op == Op.MASKED_SOFTMAX:
        logits = _read_node_input(device, node.inputs[0], layout)
        mask = _read_node_input(device, node.inputs[1], layout)
        axis = int(node.attrs.get("axis", -1))
        shifted = np.where(mask == 0, -1e30, logits)
        shifted = shifted - np.max(shifted, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        _write_node_output(
            device, output, layout, exp / np.sum(exp, axis=axis, keepdims=True)
        )
        return

    if op == Op.LAYER_NORM:
        value = _read_node_input(device, node.inputs[0], layout)
        weight = _read_node_input(device, node.inputs[1], layout)
        bias = _read_node_input(device, node.inputs[2], layout)
        eps = float(node.attrs.get("epsilon", 1e-5))
        mean = value.mean(axis=-1, keepdims=True)
        var = value.var(axis=-1, keepdims=True)
        normalized = (value - mean) / np.sqrt(var + eps)
        _write_node_output(device, output, layout, normalized * weight + bias)
        return

    if op == Op.RMS_NORM:
        value = _read_node_input(device, node.inputs[0], layout)
        weight = _read_node_input(device, node.inputs[1], layout)
        eps = float(node.attrs.get("epsilon", 1e-5))
        rms = np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + eps)
        _write_node_output(device, output, layout, value / rms * weight)
        return

    if op == Op.CAUSAL_MASK:
        batch = int(node.attrs.get("batch", 1))
        query_len = int(node.attrs.get("query_len"))
        key_len = int(node.attrs.get("key_len"))
        mask = np.tril(np.ones((query_len, key_len), dtype=np.float32))
        mask = mask.reshape(1, 1, query_len, key_len).repeat(batch, axis=0)
        _write_node_output(device, output, layout, mask)
        return

    if op == Op.ROPE:
        x = _read_node_input(device, node.inputs[0], layout)
        cos = _read_node_input(device, node.inputs[1], layout)
        sin = _read_node_input(device, node.inputs[2], layout)
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = x0 * cos - x1 * sin
        out[..., 1::2] = x0 * sin + x1 * cos
        _write_node_output(device, output, layout, out)
        return

    if op == Op.KV_CACHE_APPEND:
        cache = _read_node_input(device, node.inputs[0], layout)
        value = _read_node_input(device, node.inputs[1], layout)
        position = int(node.attrs.get("position", 0))
        cache[:, :, position, :] = value
        _write_node_output(device, output, layout, cache)
        return

    if op == Op.KV_CACHE_GATHER:
        cache = _read_node_input(device, node.inputs[0], layout)
        positions = _read_node_input(device, node.inputs[1], layout).astype(int)
        gathered = np.stack(
            [cache[:, :, positions[:, i], :] for i in range(positions.shape[1])],
            axis=2,
        )
        _write_node_output(device, output, layout, gathered)
        return

    if op == Op.EMBEDDING_GATHER:
        indices = _read_node_input(device, node.inputs[0], layout).astype(int)
        weight = _read_node_input(device, node.inputs[1], layout)
        _write_node_output(device, output, layout, weight[indices])
        return

    if op == Op.ARGMAX:
        value = _read_node_input(device, node.inputs[0], layout)
        axis = int(node.attrs.get("axis", -1))
        _write_node_output(
            device,
            output,
            layout,
            np.argmax(value, axis=axis).astype(np.int32),
        )
        return

    if op in {Op.SCALED_DOT_PRODUCT_ATTENTION, Op.MASKED_ATTENTION}:
        q = _read_node_input(device, node.inputs[0], layout)
        k = _read_node_input(device, node.inputs[1], layout)
        v = _read_node_input(device, node.inputs[2], layout)
        scale = float(node.attrs.get("scale", q.shape[-1] ** 0.5))
        scores = np.matmul(q, np.swapaxes(k, -1, -2)) / scale
        if op == Op.MASKED_ATTENTION:
            mask = _read_node_input(device, node.inputs[3], layout)
            scores = np.where(mask == 0, -1e30, scores)
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        _write_node_output(device, output, layout, np.matmul(weights, v))
        return

    raise NotImplementedError(f"no software implementation for {op.value!r}")
