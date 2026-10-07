"""Emit MLIR for a named graph used by lit tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from compiler.frontends.manual_frontend import ManualFrontend
from compiler.mlir import MLIRBuilder
from compiler.model_ir import DType


def _vector_add():
    frontend = ManualFrontend("vector_add")
    a = frontend.add_input("a", DType.F32, (8,))
    b = frontend.add_input("b", DType.F32, (8,))
    frontend.output(frontend.add(a, b))
    return frontend.build()


def _matmul():
    frontend = ManualFrontend("matmul")
    a = frontend.add_input("a", DType.F32, (2, 4))
    b = frontend.add_input("b", DType.F32, (4, 3))
    frontend.output(frontend.matmul(a, b))
    return frontend.build()


def _conv2d():
    frontend = ManualFrontend("conv2d")
    image = frontend.add_input("image", DType.F32, (1, 3, 8, 8))
    weight = frontend.add_input("weight", DType.F32, (4, 3, 3, 3))
    frontend.output(frontend.conv2d(image, weight))
    return frontend.build()


CASES = {
    "vector_add": _vector_add,
    "matmul": _matmul,
    "conv2d": _conv2d,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    args = parser.parse_args()

    graph = CASES[args.case]()
    sys.stdout.write(MLIRBuilder().build(graph))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
