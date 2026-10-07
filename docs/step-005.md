# Step 005: Unified Model IR and Manual Frontend

This document describes the first compiler milestone.

Step 005 adds a language-neutral graph representation shared by the CNN/object
detection and Transformer/LLM tracks.  It also adds a manual Python frontend
used to build, print, and validate small model graphs before ONNX and PyTorch
FX importers are introduced.

---

## 1. Model IR

File: [../compiler/model_ir.py](../compiler/model_ir.py)

Core types:

- `DType`: FP32, FP16, BF16, integer, INT4/UINT4, FP8 E4M3/E5M2, and boolean.
- `Value`: a typed SSA value with a name, dtype, and static shape.
- `Node`: an SSA operation with inputs, outputs, and attributes.
- `Graph`: inputs, nodes, and outputs with validation and printing.

The operator inventory is defined in one `Op` enum and covers:

- elementwise and activation,
- reductions,
- matmul, linear, and batch matmul,
- conv2d, depthwise conv2d, and pooling,
- normalization,
- softmax, attention, RoPE, and KV-cache,
- reshape/transpose/concat/slice/pad/resize/expand/squeeze/unsqueeze,
- object-detection helpers `box_decode` and `nms`.

Every node is created through `Graph.add_node`, which runs shape and dtype
inference immediately.

---

## 2. Manual frontend

File: [../compiler/frontends/manual_frontend.py](../compiler/frontends/manual_frontend.py)

`ManualFrontend` provides typed Python methods for the same operator set.

Example:

```python
from compiler.frontends.manual_frontend import ManualFrontend
from compiler.model_ir import DType

frontend = ManualFrontend("example")
x = frontend.add_input("x", DType.F32, (1, 4, 8, 16))
w = frontend.add_input("w", DType.F32, (16, 16))
y = frontend.linear(x, w)
frontend.output(y)

graph = frontend.build()
print(graph.format())
```

`build()` calls `Graph.validate`, so undefined inputs, malformed shapes, and
unsupported operator configurations fail before the MLIR stage.

---

## 3. Files changed

- [../compiler/__init__.py](../compiler/__init__.py)
- [../compiler/model_ir.py](../compiler/model_ir.py)
- [../compiler/frontends/__init__.py](../compiler/frontends/__init__.py)
- [../compiler/frontends/manual_frontend.py](../compiler/frontends/manual_frontend.py)
- [../tests/unit/test_model_ir.py](../tests/unit/test_model_ir.py)
- [../tests/unit/test_manual_frontend.py](../tests/unit/test_manual_frontend.py)

---

## 4. Tests

New tests cover:

- value dtype and shape metadata,
- undefined-input rejection,
- conv2d shape inference,
- graph text formatting,
- a small CNN graph with conv/pool/linear/sigmoid,
- a Transformer block with RoPE, attention, residual add, and LayerNorm,
- shape/layout operators,
- additional CNN/Transformer operators and detection helpers,
- invalid shape rejection.

Run:

```bash
source .venv/bin/activate
python -m unittest discover -v
```

All existing tests continue to pass.
