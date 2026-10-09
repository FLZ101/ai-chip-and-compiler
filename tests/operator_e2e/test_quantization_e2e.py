"""End-to-end tests for the explicit ``mgpu`` quantization ops."""

from __future__ import annotations

import unittest

import numpy as np
from ml_dtypes import float8_e4m3fn

from compiler.mlir import run_mgpu_pipeline
from hw.device import create_device, launch
from hw.isa import parse_asm


def _quant_mlir(n: int) -> str:
    return f"""
module {{
  func.func @quant(%arg0: memref<{n}xf32>) -> memref<{n}xi8> {{
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %out = memref.alloc() : memref<{n}xi8>
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c32, %c1, %c1) {{
      "mgpu.quant"(%arg0, %out) <{{scale = 5.000000e-01 : f32, zeroPoint = 0 : i32, storageType = i8}}> : (memref<{n}xf32>, memref<{n}xi8>) -> ()
      mgpu.return
    }}
    return %out : memref<{n}xi8>
  }}
}}
"""


def _dequant_mlir(n: int, storage_type: str = "i8", compute_type: str = "f32") -> str:
    return f"""
module {{
  func.func @dequant(%arg0: memref<{n}x{storage_type}>) -> memref<{n}x{compute_type}> {{
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %out = memref.alloc() : memref<{n}x{compute_type}>
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c32, %c1, %c1) {{
      "mgpu.dequant"(%arg0, %out) <{{scale = 5.000000e-01 : f32, zeroPoint = 0 : i32, computeType = {compute_type}}}> : (memref<{n}x{storage_type}>, memref<{n}x{compute_type}>) -> ()
      mgpu.return
    }}
    return %out : memref<{n}x{compute_type}>
  }}
}}
"""


def _requant_mlir(n: int) -> str:
    return f"""
module {{
  func.func @requant(%arg0: memref<{n}xi8>) -> memref<{n}xi8> {{
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %out = memref.alloc() : memref<{n}xi8>
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c32, %c1, %c1) {{
      "mgpu.requant"(%arg0, %out) <{{inputScale = 5.000000e-01 : f32, inputZeroPoint = 0 : i32, outputScale = 2.500000e-01 : f32, outputZeroPoint = 0 : i32, storageType = i8}}> : (memref<{n}xi8>, memref<{n}xi8>) -> ()
      mgpu.return
    }}
    return %out : memref<{n}xi8>
  }}
}}
"""


def _matmul_mlir(
    lhs_type: str,
    rhs_type: str,
    out_type: str,
    m: int,
    k: int,
    n: int,
) -> str:
    return f"""
module {{
  func.func @matmul(%a: memref<{m}x{k}x{lhs_type}>, %b: memref<{k}x{n}x{rhs_type}>) -> memref<{m}x{n}x{out_type}> {{
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c = memref.alloc() : memref<{m}x{n}x{out_type}>
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c32, %c1, %c1) {{
      linalg.matmul ins(%a, %b : memref<{m}x{k}x{lhs_type}>, memref<{k}x{n}x{rhs_type}>) outs(%c : memref<{m}x{n}x{out_type}>)
      mgpu.return
    }}
    return %c : memref<{m}x{n}x{out_type}>
  }}
}}
"""


def _conv_mlir() -> str:
    return """
module {
  func.func @conv(%x: memref<1x1x4x4xi8>, %w: memref<1x1x3x3xi8>) -> memref<1x1x2x2xi32> {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %out = memref.alloc() : memref<1x1x2x2xi32>
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c32, %c1, %c1) {
      linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%x, %w : memref<1x1x4x4xi8>, memref<1x1x3x3xi8>) outs(%out : memref<1x1x2x2xi32>)
      mgpu.return
    }
    return %out : memref<1x1x2x2xi32>
  }
}
"""


class TestQuantizationEndToEnd(unittest.TestCase):
    """Phase 5.1/5.2 explicit quantization op lowering."""

    def test_quantize_roundtrip(self):
        n = 8
        result = run_mgpu_pipeline(_quant_mlir(n))
        values = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], np.float32)

        device = create_device(global_size=1 << 20, sm_count=4)
        device.global_memory.load_floats(0, values.tolist())
        launch(
            device,
            parse_asm(result.code),
            grid_dim=result.grid,
            block_dim=result.block,
        )

        actual = np.asarray(
            [device.global_memory.read_i8(n * 4 + i) for i in range(n)],
            dtype=np.int8,
        )
        expected = np.round(values / 0.5).astype(np.int8)
        np.testing.assert_array_equal(actual, expected)

    def test_dequantize_roundtrip(self):
        n = 8
        result = run_mgpu_pipeline(_dequant_mlir(n))
        quantized = [-4, -2, 0, 2, 4, 6, 8, 10]

        device = create_device(global_size=1 << 20, sm_count=4)
        for index, value in enumerate(quantized):
            device.global_memory.write_i8(index, value)
        launch(
            device,
            parse_asm(result.code),
            grid_dim=result.grid,
            block_dim=result.block,
        )

        actual = np.asarray(device.global_memory.read_floats(n, n), dtype=np.float32)
        expected = np.asarray(quantized, dtype=np.float32) * 0.5
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_i4_dequantize_unpacks_nibbles(self):
        n = 8
        result = run_mgpu_pipeline(_dequant_mlir(n, storage_type="i4"))
        quantized = [-4, -2, 0, 1, 2, 3, 4, 5]

        device = create_device(global_size=1 << 20, sm_count=4)
        for index, value in enumerate(quantized):
            device.global_memory.write_i4(index, value)
        launch(
            device,
            parse_asm(result.code),
            grid_dim=result.grid,
            block_dim=result.block,
        )

        input_bytes = (n + 1) // 2
        actual = np.asarray(
            device.global_memory.read_floats(input_bytes, n), dtype=np.float32
        )
        expected = np.asarray(quantized, dtype=np.float32) * 0.5
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_requantize(self):
        n = 8
        result = run_mgpu_pipeline(_requant_mlir(n))
        quantized = [-3, -1, 0, 1, 2, 3, 4, 5]

        device = create_device(global_size=1 << 20, sm_count=4)
        for index, value in enumerate(quantized):
            device.global_memory.write_i8(index, value)
        launch(
            device,
            parse_asm(result.code),
            grid_dim=result.grid,
            block_dim=result.block,
        )

        actual = np.asarray(
            [device.global_memory.read_i8(n + i) for i in range(n)],
            dtype=np.int8,
        )
        expected = np.round(np.asarray(quantized) * (0.5 / 0.25)).astype(np.int8)
        np.testing.assert_array_equal(actual, expected)

    def test_int8_matmul_uses_i32_accumulator(self):
        m, k, n = 2, 3, 2
        result = run_mgpu_pipeline(_matmul_mlir("i8", "i8", "i32", m, k, n))
        a = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
        b = np.asarray([[7, 8], [9, 10], [11, 12]], dtype=np.int8)

        device = create_device(global_size=1 << 20, sm_count=4)
        a_bytes = m * k
        b_bytes = k * n
        for index, value in enumerate(a.reshape(-1).tolist()):
            device.global_memory.write_i8(index, value)
        for index, value in enumerate(b.reshape(-1).tolist()):
            device.global_memory.write_i8(a_bytes + index, value)

        launch(
            device,
            parse_asm(result.code),
            grid_dim=result.grid,
            block_dim=result.block,
        )

        output_base = a_bytes + b_bytes
        actual = np.asarray(
            [device.global_memory.read_i32(output_base + 4 * i) for i in range(m * n)],
            dtype=np.int32,
        ).reshape(m, n)
        expected = a.astype(np.int32) @ b.astype(np.int32)
        np.testing.assert_array_equal(actual, expected)

    def test_fp8_e4m3_matmul(self):
        m, k, n = 2, 3, 2
        result = run_mgpu_pipeline(_matmul_mlir("f8E4M3FN", "f8E4M3FN", "f32", m, k, n))
        a = np.asarray([[1.0, -0.5, 0.25], [0.125, 0.5, -0.25]], dtype=np.float32)
        b = np.asarray([[0.5, 1.0], [-0.25, 0.125], [0.0625, -0.5]], dtype=np.float32)

        device = create_device(global_size=1 << 20, sm_count=4)
        a_bytes = m * k
        b_bytes = k * n
        for index, value in enumerate(a.reshape(-1).tolist()):
            device.global_memory.write_f8_e4m3(index, value)
        for index, value in enumerate(b.reshape(-1).tolist()):
            device.global_memory.write_f8_e4m3(a_bytes + index, value)

        launch(
            device,
            parse_asm(result.code),
            grid_dim=result.grid,
            block_dim=result.block,
        )

        output_base = a_bytes + b_bytes
        actual = np.asarray(
            device.global_memory.read_floats(output_base, m * n), dtype=np.float32
        ).reshape(m, n)
        expected = np.asarray(a, dtype=float8_e4m3fn).astype(np.float32) @ np.asarray(
            b, dtype=float8_e4m3fn
        ).astype(np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_int8_conv_uses_i32_accumulator(self):
        result = run_mgpu_pipeline(_conv_mlir())
        image = np.arange(16, dtype=np.int32).reshape(1, 1, 4, 4) % 5
        weight = np.arange(9, dtype=np.int32).reshape(1, 1, 3, 3) % 3

        device = create_device(global_size=1 << 20, sm_count=4)
        for index, value in enumerate(image.reshape(-1).tolist()):
            device.global_memory.write_i8(index, int(value))
        weight_base = 16
        for index, value in enumerate(weight.reshape(-1).tolist()):
            device.global_memory.write_i8(weight_base + index, int(value))

        launch(
            device,
            parse_asm(result.code),
            grid_dim=result.grid,
            block_dim=result.block,
        )

        output_base = weight_base + 9
        actual = np.asarray(
            [device.global_memory.read_i32(output_base + 4 * i) for i in range(4)],
            dtype=np.int32,
        ).reshape(2, 2)
        expected = np.zeros((2, 2), dtype=np.int32)
        for oy in range(2):
            for ox in range(2):
                total = 0
                for ky in range(3):
                    for kx in range(3):
                        total += int(image[0, 0, oy + ky, ox + kx]) * int(
                            weight[0, 0, ky, kx]
                        )
                expected[oy, ox] = total
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
