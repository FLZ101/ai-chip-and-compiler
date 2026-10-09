#!/usr/bin/env python3
"""Compare Python and native C++ simulator wall-clock time."""

from __future__ import annotations

import os
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hw.device import create_device, launch  # noqa: E402
from hw.isa import parse_asm  # noqa: E402

ASM = """
MOV R0, ctaid.x
MUL R0, R0, ntid.x
ADD R0, R0, tid.x
MOV R20, 0
MOV R21, 0
MOV R22, 10000
loop:
CMP.LT R23, R21, R22
BRA body, R23, end
BRA end
body:
ADD R20, R20, R21
ADD R21, R21, 1
BRA loop
end:
MUL R0, R0, 4
STG [R0], R20
EXIT
"""


def run_once(*, use_cpp: bool) -> float:
    device = create_device(global_size=1 << 20, sm_count=4)
    program = parse_asm(ASM)
    if use_cpp:
        os.environ["MGPU_CPP_SIM"] = "1"
    else:
        os.environ.pop("MGPU_CPP_SIM", None)
    start = time.perf_counter()
    launch(device, program, grid_dim=(32,), block_dim=(32,))
    return time.perf_counter() - start


def main() -> None:
    python_seconds = run_once(use_cpp=False)
    cpp_seconds = run_once(use_cpp=True)
    print(f"python: {python_seconds:.6f}s")
    print(f"cpp:    {cpp_seconds:.6f}s")
    if cpp_seconds > 0:
        print(f"speedup: {python_seconds / cpp_seconds:.1f}x")


if __name__ == "__main__":
    main()
