"""Public device API.

The rest of the project should import device construction and kernel launch
from this module rather than reaching into ``hw.core`` directly.  That keeps
the execution-core implementation free to evolve behind a stable public seam.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from .core import Device, create_device, launch as _python_launch
from .timing import ProfileCounters, StallCounters, TimingModel

__all__ = [
    "Device",
    "ProfileCounters",
    "StallCounters",
    "TimingModel",
    "create_device",
    "launch",
]


def _cpp_launch(device: Device, program: object, *, grid_dim, block_dim, profile):
    """Execute a parsed program with the optional native C++ simulator."""
    build_dir = Path(__file__).resolve().parents[1] / ".deps" / "cpp-sim-build"
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    cpp_sim = importlib.import_module("cpp_sim")

    shared_size = len(device.sms[0].shared_memory.data)
    stats = cpp_sim.run(
        device.global_memory.data,
        shared_size,
        len(device.sms),
        device.register_count,
        program.source,
        tuple(int(dim) for dim in grid_dim),
        tuple(int(dim) for dim in block_dim),
        profile,
    )

    device.clock += int(stats["cycles"])
    device.profile = ProfileCounters(
        cycles=int(stats["cycles"]),
        issued_instructions=int(stats["issued_instructions"]),
        global_load_bytes=int(stats["global_load_bytes"]),
        global_store_bytes=int(stats["global_store_bytes"]),
        shared_load_bytes=int(stats["shared_load_bytes"]),
        shared_store_bytes=int(stats["shared_store_bytes"]),
        shared_bank_conflicts=int(stats["shared_bank_conflicts"]),
    )
    return device.profile.snapshot() if profile else None


def launch(
    device: Device,
    program: object,
    *,
    grid_dim=(1,),
    block_dim=(1,),
    profile: bool = False,
):
    if os.environ.get("MGPU_CPP_SIM") == "1":
        return _cpp_launch(
            device,
            program,
            grid_dim=grid_dim,
            block_dim=block_dim,
            profile=profile,
        )
    return _python_launch(
        device,
        program,
        grid_dim=grid_dim,
        block_dim=block_dim,
        profile=profile,
    )
