# `mgpu-opt`

`mgpu-opt` is the project-specific MLIR optimizer driver.

It registers the custom `mgpu` dialect and will later run the full lowering
pipeline:

```text
imported MLIR -> mgpu dialect -> simulator ISA
```

---

## 1. Location

Out-of-tree source:

```text
mlir/mgpu/
```

Default build output:

```text
.deps/mgpu-build/bin/mgpu-opt
```

Build with:

```bash
./scripts/build_mlir.sh
```

Manual build:

```bash
cmake -S mlir/mgpu -B .deps/mgpu-build -G Ninja \
  -DMLIR_DIR=.deps/llvm-build/lib/cmake/mlir \
  -DLLVM_DIR=.deps/llvm-build/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release

cmake --build .deps/mgpu-build --target mgpu-opt -j 8
```

---

## 2. Current role

Today `mgpu-opt` can:

- parse MLIR,
- register the `mgpu` dialect,
- verify custom `mgpu.*` operations,
- print the module back deterministically,
- run the `tile-linalg` pass to tile buffer-form linalg operations,
- run upstream passes such as `one-shot-bufferize`.

The remaining project-specific lowering passes are implemented in later phases.

---

## 3. Supported `mgpu` operations

The out-of-tree dialect currently supports:

- `mgpu.launch`
- `mgpu.thread_id`
- `mgpu.block_id`
- `mgpu.grid_dim`
- `mgpu.block_dim`
- `mgpu.shared_alloc`
- `mgpu.barrier`
- `mgpu.async_copy`
- `mgpu.mma`
- `mgpu.return`

Dimension operands currently use `I32Attr` rather than symbolic enum attributes.

Example:

```mlir
module {
  func.func @example() {
    %c1 = arith.constant 1 : index
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) {
      %tx = mgpu.thread_id 0 : index
      %bx = mgpu.block_id 0 : index
      %gx = mgpu.grid_dim 0 : index
      %bd = mgpu.block_dim 0 : index
      mgpu.barrier
      mgpu.return
    }
    return
  }
}
```

---

## 4. Command-line usage

Parse and print:

```bash
mgpu-opt input.mlir -o output.mlir
```

Run canonicalization later:

```bash
mgpu-opt input.mlir \
  --pass-pipeline="builtin.module(canonicalize,cse)" \
  -o canonical.mlir
```

Run the implemented linalg tiling pass:

```bash
mgpu-opt input.mlir \
  --pass-pipeline="builtin.module(func.func(tile-linalg))" \
  -o tiled.mlir
```

Run the future full pipeline:

```bash
mgpu-opt input.mlir \
  --pass-pipeline="builtin.module(
    model-to-linalg,
    tile-linalg,
    one-shot-bufferize,
    map-to-mgpu,
    lower-to-mgpu,
    mgpu-to-isa
  )" \
  -o output.isa
```

---

## 5. Future lowering passes

The following passes will be implemented inside or invoked by `mgpu-opt`:

- `model-to-linalg`
- `tile-linalg`
- `one-shot-bufferize`
- `map-to-mgpu`
- `lower-to-mgpu`
- `mgpu-to-isa`

Each pass should have FileCheck tests in `tests/mlir/`.

---

## 6. Testing

Manual smoke test:

```bash
cat > /tmp/mgpu_min.mlir <<'EOF'
module {
  func.func @foo() {
    %c1 = arith.constant 1 : index
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) {
      %tx = mgpu.thread_id 0 : index
      mgpu.barrier
      mgpu.return
    }
    return
  }
}
EOF

mgpu-opt /tmp/mgpu_min.mlir -o /tmp/mgpu_min.out.mlir
```

Expected result: the module parses and prints successfully.

---

## 7. Current limitations

- No lowering passes yet except `tile-linalg`.
- Dimension attributes are numeric `I32Attr`, not symbolic `x`/`y`/`z`.
- `mgpu.shared_alloc`, `mgpu.async_copy`, and `mgpu.mma` are defined but not
  lowered or executed yet.
- Python still generates MLIR text; `mgpu-opt` is not yet called by the driver.
