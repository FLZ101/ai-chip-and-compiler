# Step 2.1: Real PyTorch FX Import with torch-mlir

This document describes the implementation of Phase 2.1 from
[next-stage-steps.md](next-stage-steps.md).

The goal is to stop relying on the original hand-written FX-to-Model-IR walker
for the production path and instead import real PyTorch FX programs with
`torch-mlir`.

---

## 1. What was implemented

Added [../compiler/mlir/importer.py](../compiler/mlir/importer.py), a small
frontend that:

1. exports an `nn.Module` with `torch.export.export`,
2. imports the resulting `ExportedProgram` with
   `torch_mlir.fx.export_and_import`,
3. returns stable MLIR text plus a live `torch_mlir.ir.Module`,
4. preserves the original PyTorch parameter/buffer/constant names from the
   exported graph signature.

The old manual `TorchFXFrontend.convert()` path is intentionally retained for
the existing simulator E2E tests. The new production entry point is
`TorchFXFrontend.import_mlir()`.

---

## 2. API

```python
from compiler.mlir import TorchMLIRImporter

result = TorchMLIRImporter().import_module(
    model,
    torch.zeros(1, 3, 224, 224),
    func_name="resnet_block",
)

print(result.text)            # stable MLIR module text
print(result.parameter_map)   # lifted placeholder -> original parameter name
```

`result.mlir_module.operation.verify()` returns `True` for every imported
module, and the text can be round-tripped through MLIR parse/print.

---

## 3. Parameter and buffer preservation

`torch.export` records the relationship between lifted placeholders and the
original state-dict names in:

- `graph_signature.inputs_to_parameters`
- `graph_signature.inputs_to_buffers`
- `graph_signature.inputs_to_lifted_tensor_constants`

The importer copies those mappings into:

- `TorchMLIRImportResult.parameter_map`
- `TorchMLIRImportResult.buffer_map`
- `TorchMLIRImportResult.constant_map`

For example, a module with `LayerNorm` and `Linear` produces:

```text
{
  "p_norm_weight": "norm.weight",
  "p_norm_bias": "norm.bias",
  "p_proj_weight": "proj.weight",
  "p_proj_bias": "proj.bias",
}
```

---

## 4. Determinism

For a fixed module and fixed example inputs, importing twice yields identical
MLIR text, including the `dense_resource` payload section. This is verified by
[../tests/unit/test_torch_mlir_importer.py](../tests/unit/test_torch_mlir_importer.py).

---

## 5. Tests

New tests:

- small CNN import,
- small Transformer block import,
- deterministic repeated import,
- `TorchFXFrontend.import_mlir()` integration.

Run:

```bash
source .venv/bin/activate
python -m unittest -v tests.unit.test_torch_mlir_importer
```

---

## 6. Files changed

- [../compiler/mlir/importer.py](../compiler/mlir/importer.py)
- [../compiler/mlir/__init__.py](../compiler/mlir/__init__.py)
- [../compiler/frontends/torch_fx_frontend.py](../compiler/frontends/torch_fx_frontend.py)
- [../tests/unit/test_torch_mlir_importer.py](../tests/unit/test_torch_mlir_importer.py)
