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
from dataclasses import dataclass
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
    alignment: int = 4096,
) -> MemoryLayout:
    """Write host tensors into global memory and return their placements."""
    layout: MemoryLayout = {}
    cursor = 0

    for name, array in tensors.items():
        if cursor > 0:
            cursor = ((cursor + alignment - 1) // alignment) * alignment

        data = np.asarray(array, dtype=np.float32)
        base_addr = cursor
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

    intermediate_tensors = {
        value.name: np.zeros(value.shape, dtype=np.float32)
        for node in graph.nodes
        for value in node.outputs
    }
    layout = allocate_global_buffers(device, {**inputs, **intermediate_tensors})

    if len(graph.nodes) == 1:
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
        for node in graph.nodes:
            subgraph = Graph(
                name=node.name,
                inputs=list(node.inputs),
                outputs=list(node.outputs),
                nodes=[node],
            )
            subgraph.validate()
            node_isa, (grid_dim, block_dim) = lowering(subgraph, mlir_text, layout)
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
