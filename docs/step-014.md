# Step 014: ONNX and PyTorch FX Frontends

This document describes the model-import frontends.

Step 014 adds:

- `compiler/frontends/onnx_frontend.py`
- `compiler/frontends/torch_fx_frontend.py`

Both frontends build the same unified Model IR used by the manual frontend.

---

## 1. ONNX frontend

File: [../compiler/frontends/onnx_frontend.py](../compiler/frontends/onnx_frontend.py)

The ONNX importer supports the small CNN subset needed by early tests:

- `Conv`,
- `Relu`,
- `MaxPool`,
- `Add`,
- `Flatten`.

ONNX initializers are represented as graph inputs.  Dynamic dimensions are
rejected because the current Model IR uses static shapes.

---

## 2. PyTorch FX frontend

File: [../compiler/frontends/torch_fx_frontend.py](../compiler/frontends/torch_fx_frontend.py)

The FX importer supports small Transformer blocks containing:

- `nn.LayerNorm`,
- `nn.Linear`,
- `nn.ReLU`,
- `torch.relu`,
- `torch.nn.functional.softmax`,
- `torch.add`,
- `reshape`/`transpose` methods.

Placeholder shapes are supplied explicitly:

```python
TorchFXFrontend(
    "fx_model",
    input_shapes={"x": ((2, 4), DType.F32)},
)
```

Module parameters are added as graph inputs.

---

## 3. MLIR builder update

[../compiler/mlir/mlir_builder.py](../compiler/mlir/mlir_builder.py) now emits
placeholder MLIR for `linear`, so FX Transformer blocks can lower through the
same `Model IR -> MLIR` path.

---

## 4. Files changed

- [../compiler/frontends/onnx_frontend.py](../compiler/frontends/onnx_frontend.py)
- [../compiler/frontends/torch_fx_frontend.py](../compiler/frontends/torch_fx_frontend.py)
- [../compiler/frontends/__init__.py](../compiler/frontends/__init__.py)
- [../compiler/mlir/mlir_builder.py](../compiler/mlir/mlir_builder.py)
- [../compiler/codegen/activation_kernels.py](../compiler/codegen/activation_kernels.py)
- [../tests/unit/test_onnx_frontend.py](../tests/unit/test_onnx_frontend.py)
- [../tests/unit/test_torch_fx_frontend.py](../tests/unit/test_torch_fx_frontend.py)

---

## 5. Tests

New tests cover:

- ONNX CNN import, MLIR lowering, and simulator E2E execution,
- PyTorch FX Transformer block import and MLIR lowering.

Run:

```bash
source .venv/bin/activate
python -m unittest discover -v
```

All existing tests continue to pass.
