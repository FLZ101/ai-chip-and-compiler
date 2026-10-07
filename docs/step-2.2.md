# Step 2.2: Real ONNX Import with torch-mlir

This document describes the implementation of Phase 2.2 from
[next-stage-steps.md](next-stage-steps.md).

The production path now imports ONNX models directly with torch-mlir instead
of walking `onnx.NodeProto` items into the small manual Model IR.

---

## 1. What was implemented

Added an ONNX importer to [../compiler/mlir/importer.py](../compiler/mlir/importer.py):

- runs `onnx.shape_inference.infer_shapes`,
- registers the torch dialect in a real MLIR context,
- uses `torch_mlir.extras.onnx_importer` to import the whole graph,
- verifies the resulting module,
- returns stable MLIR text plus the original ONNX metadata.

The importer deliberately uses torch-mlir's `ModelInfo` and `NodeImporter`
rather than a custom op-by-op translator.

---

## 2. ONNX frontend integration

[../compiler/frontends/onnx_frontend.py](../compiler/frontends/onnx_frontend.py)
now exposes:

```python
result = ONNXFrontend("onnx_model").import_mlir(model_proto)
```

This is the Phase 2.2 entry point. The older manual `convert()` path remains
available for existing simulator E2E tests.

---

## 3. Initializers

The importer preserves every ONNX initializer name:

```python
result.initializer_names
# ("conv_weight", "conv_bias", ...)
```

Torch-mlir's default `elide_initialized_inputs` behavior turns initializers
into `torch.operator "onnx.Constant"` ops in the MLIR function body.

---

## 4. Dynamic dimensions

The importer records non-initializer graph inputs and their ONNX dimensions:

```python
result.dynamic_inputs
# {"x": ("batch", 4)}
```

Static dimensions become `int`, dynamic `dim_param` values remain `str`, and
unknown dimensions become `None`. In the MLIR text, dynamic dimensions import
as `!torch.vtensor<[?,4],f32>`.

---

## 5. Tests

New tests in
[../tests/unit/test_onnx_mlir_importer.py](../tests/unit/test_onnx_mlir_importer.py)
cover:

- CNN import without manual mapping,
- dynamic dimension placeholder preservation,
- Transformer-style block import through `ONNXFrontend.import_mlir`.

Run:

```bash
source .venv/bin/activate
python -m unittest -v tests.unit.test_onnx_mlir_importer
```

---

## 6. Files changed

- [../compiler/mlir/importer.py](../compiler/mlir/importer.py)
- [../compiler/mlir/__init__.py](../compiler/mlir/__init__.py)
- [../compiler/frontends/onnx_frontend.py](../compiler/frontends/onnx_frontend.py)
- [../tests/unit/test_onnx_mlir_importer.py](../tests/unit/test_onnx_mlir_importer.py)
