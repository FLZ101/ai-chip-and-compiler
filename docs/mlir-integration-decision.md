There are three practical integration models.

## 1. Native C++ MLIR/LLVM toolchain

You build against LLVM/MLIR C++ libraries and create your own tools:

```text
mgpu-opt
mgpu-translate
libMGPU
```

Advantages:

- Full MLIR API and all passes.
- Best performance.
- Standard production path used by MLIR-based compilers.
- Can use `mlir-tblgen` directly from `.td`.
- Easier to integrate custom C++ transforms, allocators, and runtime drivers.

Disadvantages:

- Heavy build and dependency management.
- Requires CMake/LLVM version alignment.
- Slower iteration for small experiments.
- Packaging a Python wheel is more work.

Best when:

- The project is becoming a real compiler.
- Passes need low-level C++ control.
- Performance of the compiler itself matters.

## 2. Python MLIR bindings / torch-mlir

Use installed Python bindings:

```text
torch-mlir
mlir-python-bindings
```

You write passes either:

- in Python where supported,
- or call prebuilt MLIR tools from Python.

Advantages:

- Fast iteration.
- Very good for import and frontend work.
- `torch-mlir` already includes PyTorch FX and ONNX importers.
- No custom CMake build.
- Easy to run in the current project and tests.

Disadvantages:

- Not all MLIR APIs are exposed in Python.
- Custom C++ passes cannot be written directly.
- Pass execution may be slower.
- Dialect registration for custom `mgpu` ops is harder unless provided by a compiled extension.
- Long-term, Python bindings may limit advanced lowering work.

Best when:

- The immediate goal is importing models and prototyping passes.
- The project is still defining the IR and pass pipeline.

## 3. Hybrid model

This is usually the best choice for this project.

Use Python/torch-mlir for:

- PyTorch FX import.
- ONNX import.
- early canonicalization.
- tests and rapid iteration.

Use a small compiled C++ `mgpu-opt` for:

- registering the custom `mgpu` dialect.
- running project-specific passes.
- final `mgpu-to-isa` lowering.
- FileCheck/lit testing.

Flow:

```text
PyTorch/ONNX
   -> torch-mlir Python import
   -> MLIR text or bytecode
   -> mgpu-opt C++ tool
   -> mgpu dialect
   -> ISA
```

## Decision factors

| Factor | Python bindings | C++ toolchain | Hybrid |
| --- | --- | --- | --- |
| Setup cost | low | high | medium |
| Import speed | fast | slower | fast |
| Custom dialect | awkward | natural | natural |
| Custom passes | limited | full | full |
| Compiler performance | slower | fast | fast where needed |
| Long-term suitability | weaker | strongest | strong |
| Best use | frontend/prototyping | production passes | recommended path |

## Recommendation

Choose the **hybrid model**:

1. Keep PyTorch FX and ONNX import in Python via `torch-mlir`.
2. Build a small C++ `mgpu-opt` that registers the `mgpu` dialect and runs project passes.
3. Use `lit` + FileCheck against `mgpu-opt`.
4. Later, if needed, expose `mgpu-opt` as a Python subprocess or embed it behind the current Python driver.

This gives fast frontend work without sacrificing the real MLIR compiler core.
