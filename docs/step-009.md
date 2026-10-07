# Step 009: Activation and Shape/Layout Kernels

This document describes the activation and shape/layout kernel milestone.

Step 009 adds:

- `ReLU`, `LeakyReLU`, `SiLU`, `GELU`, and `Sigmoid`;
- `Reshape`, `Transpose`, `Concat`, `Slice`, `Pad`, and `Resize/Upsample`.

Each operator is exercised through the shared E2E harness and compared with a
NumPy golden result.

---

## 1. Activation kernels

File: [../compiler/codegen/activation_kernels.py](../compiler/codegen/activation_kernels.py)

Activation lowerings are unary elementwise kernels:

```text
global_tid = ctaid.x * ntid.x + tid.x
```

Implementation notes:

- `ReLU` uses `MAX`,
- `LeakyReLU` uses `CMP.LT`, `MUL`, and `SEL`,
- `Sigmoid` and `SiLU` use `EXP2`, `RCP`, and `MUL`,
- `GELU` uses the common `tanh` approximation so the simulator and NumPy
  golden reference use the same formula.

---

## 2. Shape/layout kernels

File: [../compiler/codegen/shape_kernels.py](../compiler/codegen/shape_kernels.py)

The initial shape lowerings support:

- rank-any `Reshape` as a linear copy,
- rank-2 `Transpose` using a 2D thread block,
- rank-1 `Concat` with two inputs,
- rank-1 `Slice`,
- rank-1 `Pad` with branchless `CMP.GE`/`CMP.LT`/`SEL`,
- rank-1 nearest-neighbor `Resize`.

---

## 3. ISA addition

The resize lowering needs integer floor division, so the ISA and simulator now
support:

```asm
IDIV Rd, Ra, Rb
```

`IDIV` computes `Ra // Rb` and is treated as a normal ALU instruction.

---

## 4. MLIR builder update

[../compiler/mlir/mlir_builder.py](../compiler/mlir/mlir_builder.py) now emits
`linalg.generic` text for activation ops.  Shape/layout nodes emit a clearly
marked placeholder identity `linalg.generic`; their executable lowering is the
ISA shape-kernel path.

---

## 5. Files changed

- [../compiler/codegen/activation_kernels.py](../compiler/codegen/activation_kernels.py)
- [../compiler/codegen/shape_kernels.py](../compiler/codegen/shape_kernels.py)
- [../compiler/codegen/dispatcher.py](../compiler/codegen/dispatcher.py)
- [../compiler/codegen/__init__.py](../compiler/codegen/__init__.py)
- [../compiler/mlir/mlir_builder.py](../compiler/mlir/mlir_builder.py)
- [../hw/isa.py](../hw/isa.py)
- [../hw/core.py](../hw/core.py)
- [../tests/operator_e2e/test_activation_shape_kernels_e2e.py](../tests/operator_e2e/test_activation_shape_kernels_e2e.py)
- [../tests/unit/test_isa.py](../tests/unit/test_isa.py)

---

## 6. Tests

New tests cover all activation and shape/layout operators listed above.

Run:

```bash
source .venv/bin/activate
python -m unittest discover -v
```

All existing tests continue to pass.
