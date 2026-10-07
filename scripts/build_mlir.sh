#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR="${DEPS_DIR:-${ROOT_DIR}/.deps}"
LLVM_TAG="${LLVM_TAG:-llvmorg-21.1.0}"
LLVM_SRC_DIR="${LLVM_SRC_DIR:-${DEPS_DIR}/llvm-project}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${DEPS_DIR}/llvm-build}"
MGPU_SRC_DIR="${MGPU_SRC_DIR:-${ROOT_DIR}/mlir/mgpu}"
MGPU_BUILD_DIR="${MGPU_BUILD_DIR:-${DEPS_DIR}/mgpu-build}"
JOBS="${JOBS:-8}"

mkdir -p "$DEPS_DIR"

if [[ ! -d "${LLVM_SRC_DIR}/.git" ]]; then
  echo "Cloning llvm-project ${LLVM_TAG}..."
  git clone --depth 1 --branch "$LLVM_TAG" \
    https://github.com/llvm/llvm-project.git "$LLVM_SRC_DIR"
fi

if [[ ! -x "${LLVM_BUILD_DIR}/bin/mlir-opt" ]]; then
  echo "Configuring MLIR..."
  cmake -S "${LLVM_SRC_DIR}/llvm" -B "$LLVM_BUILD_DIR" -G Ninja \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF

  echo "Building mlir-opt and mlir-tblgen..."
  cmake --build "$LLVM_BUILD_DIR" --target mlir-opt mlir-tblgen -j "$JOBS"
fi

echo "Configuring out-of-tree mgpu..."
cmake -S "$MGPU_SRC_DIR" -B "$MGPU_BUILD_DIR" -G Ninja \
  -DMLIR_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir" \
  -DLLVM_DIR="${LLVM_BUILD_DIR}/lib/cmake/llvm" \
  -DCMAKE_BUILD_TYPE=Release

echo "Building mgpu-opt..."
cmake --build "$MGPU_BUILD_DIR" --target mgpu-opt -j "$JOBS"

cat <<EOF

Build complete:
  mlir-opt:    ${LLVM_BUILD_DIR}/bin/mlir-opt
  mlir-tblgen: ${LLVM_BUILD_DIR}/bin/mlir-tblgen
  mgpu-opt:    ${MGPU_BUILD_DIR}/bin/mgpu-opt
EOF
