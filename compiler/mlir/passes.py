"""Pass definitions for the MLIR lowering pipeline.

Step 006 establishes the pass names and order.  The transformations themselves
are added in later steps:

* ``model-to-linalg`` is the MLIR builder implemented in this step,
* ``tile-linalg`` introduces shared-memory tiling,
* ``linalg-to-mgpu`` lowers tiled ops into ``mgpu.shared_alloc``,
  ``mgpu.async_copy``, and ``mgpu.mma``,
* ``mgpu-to-isa`` emits the simulator ISA.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pass:
    """One named pass in the compiler pipeline."""

    name: str
    description: str


@dataclass(frozen=True)
class PassPipeline:
    """An ordered list of passes."""

    passes: tuple[Pass, ...]

    def as_list(self) -> list[str]:
        return [pass_.name for pass_ in self.passes]

    def describe(self) -> str:
        return " -> ".join(self.as_list())


DEFAULT_PASS_PIPELINE = PassPipeline(
    passes=(
        Pass(
            "model-to-linalg",
            "Lower the unified Model IR to func/memref/linalg plus mgpu.launch.",
        ),
        Pass(
            "tile-linalg",
            "Tile matmul/conv reductions for shared-memory and register reuse.",
        ),
        Pass(
            "linalg-to-mgpu",
            "Lower tiled linalg ops to mgpu.shared_alloc, async_copy, barrier, and mma.",
        ),
        Pass(
            "mgpu-to-isa",
            "Lower the mgpu dialect to the simulator ISA and schedule warps.",
        ),
    )
)
