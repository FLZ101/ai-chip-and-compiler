# ONNX Tensor Types: FP8, INT4, and Related Formats

This document describes the tensor element types used by ONNX, with emphasis on low-precision formats relevant to CNN and Transformer/LLM inference:

- FP8 floating-point formats,
- INT4 and UINT4 packed integer formats,
- common float and integer types,
- emerging microscaling/float4 formats,
- how these types map to our simulator and compiler.

The MVP simulator currently executes FP32. This document defines the type model the frontend and compiler will consume from ONNX before lowering to the GPU ISA.

---

## 1. ONNX tensor type overview

ONNX tensors are typed by an element data type. Common families:

| Family | ONNX type names |
| --- | --- |
| IEEE-style floats | `FLOAT16`, `FLOAT32`, `FLOAT64`, `BFLOAT16` |
| FP8 | `FLOAT8E4M3FN`, `FLOAT8E4M3FNUZ`, `FLOAT8E5M2`, `FLOAT8E5M2FNUZ` |
| Signed integers | `INT4`, `INT8`, `INT16`, `INT32`, `INT64` |
| Unsigned integers | `UINT4`, `UINT8`, `UINT16`, `UINT32`, `UINT64` |
| Boolean | `BOOL` |
| Complex | `COMPLEX64`, `COMPLEX128` |
| Strings | `STRING` |
| Emerging formats | `FLOAT4E2M1`, `E8M0`, sub-byte `INT2`/`UINT2` where supported |

Exact enum values depend on the ONNX version. Compiler code should use symbolic names rather than hard-coded numbers.

---

## 2. Standard floating-point types

### FLOAT32

- 32 bits: 1 sign, 8 exponent, 23 mantissa.
- Default compute type in the current simulator.
- Used for golden references and first compiler milestone.

### FLOAT16

- 16 bits: 1 sign, 5 exponent, 10 mantissa.
- Useful for compute-bound inference and reduced bandwidth.

### BFLOAT16

- 16 bits: 1 sign, 8 exponent, 7 mantissa.
- Same dynamic range as FP32, less precision.
- Common in Transformer training and inference.

### FLOAT64

- 64-bit double precision.
- Rarely used for neural-network inference; useful for numerical tests and reference math.

---

## 3. FP8 formats

ONNX defines four FP8 types:

| ONNX type | Bits | Exponent bias | Special behavior |
| --- | --- | --- | --- |
| `FLOAT8E4M3FN` | 1S + 4E + 3M | 7 | finite-only, has infinities and NaNs |
| `FLOAT8E4M3FNUZ` | 1S + 4E + 3M | 8 | no negative zero, one NaN, finite-only |
| `FLOAT8E5M2` | 1S + 5E + 2M | 15 | has infinities and NaNs |
| `FLOAT8E5M2FNUZ` | 1S + 5E + 2M | 16 | no negative zero, one NaN, finite-only |

### E4M3FN

- 4 exponent bits and 3 mantissa bits.
- Bias is 7.
- Mainly used for weights and activations because it has more precision.
- Does not represent infinities.
- NaN is `S.1111.111`.
- Maximum normal value is approximately `1.75 * 2^15 = 57344`.

### E5M2

- 5 exponent bits and 2 mantissa bits.
- Bias is 15.
- Has a larger dynamic range but less precision.
- Often used for gradients in training; in inference it may be used for activations.

### FNUZ variants

`FNUZ` means finite-only and no unsigned zero:

- there is only one zero,
- there is only one NaN,
- exponent bias is one greater than the corresponding `FN` format.

The lack of negative zero and multiple NaN encodings simplifies some hardware implementations.

### Casting to and from FP8

ONNX `Cast` rules:

- FP8 to FP16/BF16/FP32 is exact because the higher-precision formats can represent all FP8 values.
- FP32/FP16/BF16 to FP8 requires rounding and often saturation.
- Default rounding is round-to-nearest-even.
- With saturation, out-of-range values become the largest finite FP8 value.
- Without saturation, out-of-range values may become NaN or infinity depending on the target format.

---

## 4. INT4 and UINT4

ONNX introduced 4-bit integer types primarily for weight compression in LLMs:

- `UINT4`: unsigned 4-bit integer, range `[0, 15]`.
- `INT4`: signed 4-bit integer, two's complement, range `[-8, 7]`.

### Packing

Four-bit tensors are packed two elements per byte:

```text
byte = (second_element << 4) | (first_element & 0x0F)
```

The first element is stored in the 4 LSBs. The second element is stored in the 4 MSBs.

Unpacking:

```text
first  = byte & 0x0F
second = byte >> 4
```

If a tensor has an odd number of elements, the final byte includes 4 padding bits.

For `N` elements, storage size is:

```text
ceil(N / 2) bytes
```

### Casting

- INT4/UINT4 to a wider integer or float type is exact.
- Casting from a wider type to INT4/UINT4 rounds to nearest integer, ties to even, then truncates to the representable range.

### Compiler implications

INT4/UINT4 are storage/compression types. In a GPU kernel:

1. Load packed bytes from global memory.
2. Unpack into wider integers.
3. Dequantize with scale and optional zero point.
4. Perform arithmetic in FP16/BF16/FP32.
5. Optionally quantize and pack results before storing.

For the minimal GPU, the first low-precision path should be FP16/BF16 compute. INT4 support should start as an unpack-and-dequantize path, not native 4-bit ALU operations.

---

## 5. Other integer and boolean types

### Signed integers

- `INT8`: range `[-128, 127]`.
- `INT16`: range `[-32768, 32767]`.
- `INT32`: common for shape arithmetic, indices, and accumulation.
- `INT64`: common for large tensors, shapes, and ONNX attributes.

### Unsigned integers

- `UINT8`: range `[0, 255]`.
- `UINT16`, `UINT32`, `UINT64`: wider unsigned storage.

### BOOL

- One byte per element in ONNX's standard representation.
- Used by masks, comparisons, and `Where`/`NonZero`-style operations.

---

## 6. Emerging formats: microscaling and sub-byte types

### FLOAT4E2M1

Some ONNX versions expose `FLOAT4E2M1`, a 4-bit floating-point format:

- 1 sign bit,
- 2 exponent bits,
- 1 mantissa bit.

It is extremely low precision and is intended for block-scaled/microscaling formats rather than standalone compute.

### E8M0

`E8M0` is an 8-bit exponent-only scale format:

- no sign bit,
- no mantissa bits,
- 8 exponent bits,
- exponent bias 127.

It is used as the shared scale in microscaling block formats.

### INT2 and UINT2

When supported, 2-bit integers are packed four elements per byte, from LSB to MSB.

These formats are still uncommon in mainstream inference. The compiler should treat them as optional and reject them cleanly until supported.

---

## 7. Quantization: storage type vs compute type

ONNX separates:

- **storage type**: how a tensor is stored in memory,
- **compute type**: the arithmetic type used by an operator.

This appears in the ONNX Quantize/Dequantize pattern:

```text
float32 weights
      |
   QuantizeLinear
      |
   int8/int4/uint8 tensor
      |
      +--> MatMul/Conv with optional dequantization
```

Common low-precision paths:

| Weights | Activations | Compute | Typical use |
| --- | --- | --- | --- |
| FP32 | FP32 | FP32 | golden/reference |
| FP16/BF16 | FP16/BF16 | FP16/BF16 | memory/bandwidth savings |
| INT8 | INT8 | INT32 accumulator | CNN/edge inference |
| INT4/UINT4 | FP16/FP32 | FP16/FP32 | LLM weight-only quantization |
| FP8 | FP8 | FP16/FP32 accumulator | Transformer inference |

The simulator and compiler should model these as two separate properties on every tensor: `storage_dtype` and `compute_dtype`.

---

## 8. Mapping to the minimal GPU

Current MVP:

- supports only FP32 compute and FP32 memory.

Planned type progression:

1. FP32 first.
2. Add FP16/BF16 storage and compute.
3. Add INT8 quantization path with FP32/INT32 accumulator.
4. Add INT4/UINT4 unpack-and-dequantize path for Transformer weights.
5. Add FP8 E4M3/E5M2 for matrix-heavy kernels.
6. Treat microscaling/float4 as optional future work.

ISA implications:

- memory operands remain byte-addressed;
- FP16/BF16 loads and stores become 2-byte accesses;
- INT4 loads are packed byte accesses followed by unpacking;
- FP8 loads are 1-byte accesses followed by conversion;
- arithmetic remains in a wider compute type for the MVP.

MLIR/Model IR implications:

- tensor values need explicit storage dtype and compute dtype;
- quantized models need explicit `QuantizeLinear`/`DequantizeLinear` handling;
- kernel templates must choose packing/unpacking based on storage dtype;
- the backend should reject unsupported dtype/operator combinations with a clear diagnostic.

---

## 9. Summary table

| ONNX type | Storage bits | Signed | Float | Packing | Simulator priority |
| --- | --- | --- | --- | --- | --- |
| `FLOAT32` | 32 | yes | yes | none | first-class |
| `FLOAT16` | 16 | yes | yes | none | near-term |
| `BFLOAT16` | 16 | yes | yes | none | near-term |
| `FLOAT8E4M3FN` | 8 | yes | yes | none | later |
| `FLOAT8E4M3FNUZ` | 8 | yes | yes | none | later |
| `FLOAT8E5M2` | 8 | yes | yes | none | later |
| `FLOAT8E5M2FNUZ` | 8 | yes | yes | none | later |
| `INT8` | 8 | yes | no | none | near-term |
| `UINT8` | 8 | no | no | none | optional |
| `INT4` | 4 | yes | no | 2 per byte | later |
| `UINT4` | 4 | no | no | 2 per byte | later |
| `INT32` | 32 | yes | no | none | shape/accumulator |
| `INT64` | 64 | yes | no | none | metadata/shapes |
| `BOOL` | 8 | no | no | none | masks/control |
| `FLOAT4E2M1` | 4 | yes | yes | 2 per byte | optional/future |
| `E8M0` | 8 | n/a | scale only | none | optional/future |
