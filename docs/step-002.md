# Step 002: Branches, Divergence, and Reconvergence

This document describes the second simulator milestone.

Step 002 adds SIMT control flow to the first vertical slice:

- symbolic branch labels,
- `BRA` with unconditional and per-lane conditional forms,
- `CMP` and `CMP.*` predicate generation,
- `SEL` predicated selection,
- a simple per-warp reconvergence stack.

---

## 1. ISA additions

### Labels

The parser now accepts labels:

```asm
else:
end:
```

Labels are resolved to instruction indices after parsing.

### Branch instructions

```asm
BRA target
BRA target, Rcondition
BRA target, Rcondition, reconvergence_label
```

The three-operand form is required when a branch diverges. The explicit
reconvergence label tells the simulator where both control-flow sides join.

### Compare instructions

```asm
CMP.EQ Rd, Ra, Rb
CMP.NE Rd, Ra, Rb
CMP.LT Rd, Ra, Rb
CMP.LE Rd, Ra, Rb
CMP.GT Rd, Ra, Rb
CMP.GE Rd, Ra, Rb
```

Each lane writes `1.0` when the comparison is true and `0.0` otherwise.

### Select instruction

```asm
SEL Rd, Rcondition, Rtrue, Rfalse
```

Each lane copies `Rtrue` or `Rfalse` into `Rd` based on its condition value.

---

## 2. SIMT divergence semantics

When a conditional `BRA` has both taken and fallthrough lanes:

1. Compute a `taken_mask` from active lanes whose condition is nonzero.
2. Compute `fallthrough_mask = active_mask & ~taken_mask`.
3. Push a `ReconvergenceState` containing:
   - the fallthrough start PC,
   - the fallthrough mask,
   - the reconvergence PC,
   - the original active mask.
4. Execute the taken side first.
5. When the taken side reaches the reconvergence label, switch to the saved fallthrough side.
6. When the fallthrough side also reaches the reconvergence label, pop the stack and restore the original mask.

Inactive lanes do not update registers or memory because execution only iterates over `warp.active_lanes`.

---

## 3. Files changed

- [../hw/isa.py](../hw/isa.py)
- [../hw/core.py](../hw/core.py)
- [../hw/__init__.py](../hw/__init__.py)
- [../tests/unit/test_isa.py](../tests/unit/test_isa.py)
- [../tests/operator_e2e/test_divergence_e2e.py](../tests/operator_e2e/test_divergence_e2e.py)

---

## 4. Tests

New tests cover:

- conditional branch parsing,
- undefined branch label rejection,
- simple if/else divergence,
- nested divergence,
- deterministic execution across repeated launches.

Run:

```bash
source .venv/bin/activate
python -m unittest discover -v
```

All existing tests continue to pass.
