import unittest
from pathlib import Path

from compiler.frontends.manual_frontend import ManualFrontend
from compiler.mlir import MLIRBuilder, MLIRBuilderError
from compiler.model_ir import DType


class TestMLIRBuilder(unittest.TestCase):
    """Step 006 MLIR text emission tests."""

    def _vector_add_graph(self):
        frontend = ManualFrontend("vector_add")
        a = frontend.add_input("a", DType.F32, (8,))
        b = frontend.add_input("b", DType.F32, (8,))
        output = frontend.add(a, b)
        frontend.output(output)
        return frontend.build()

    def _matmul_graph(self):
        frontend = ManualFrontend("matmul")
        a = frontend.add_input("a", DType.F32, (2, 4))
        b = frontend.add_input("b", DType.F32, (4, 3))
        output = frontend.matmul(a, b)
        frontend.output(output)
        return frontend.build()

    def _conv_graph(self):
        frontend = ManualFrontend("conv2d")
        image = frontend.add_input("image", DType.F32, (1, 3, 8, 8))
        weight = frontend.add_input("weight", DType.F32, (4, 3, 3, 3))
        output = frontend.conv2d(image, weight)
        frontend.output(output)
        return frontend.build()

    def test_vector_add_emits_deterministic_mlir(self):
        builder = MLIRBuilder()
        first = builder.build(self._vector_add_graph())
        second = builder.build(self._vector_add_graph())

        self.assertEqual(first, second)
        self.assertIn("func.func @vector_add", first)
        self.assertIn("mgpu.launch", first)
        self.assertIn("mgpu.thread_id 0 : index", first)
        self.assertIn("mgpu.block_id 0 : index", first)
        self.assertIn("mgpu.grid_dim 0 : index", first)
        self.assertIn("mgpu.block_dim 0 : index", first)
        self.assertIn("linalg.generic", first)
        self.assertIn("arith.addf", first)
        self.assertIn("mgpu.barrier", first)

    def test_matmul_emits_linalg_matmul(self):
        text = MLIRBuilder().build(self._matmul_graph())

        self.assertIn("linalg.matmul", text)
        self.assertIn("memref<2x4xf32>", text)
        self.assertIn("memref<4x3xf32>", text)
        self.assertIn("memref<2x3xf32>", text)

    def test_conv2d_emits_linalg_conv(self):
        text = MLIRBuilder().build(self._conv_graph())

        self.assertIn("linalg.conv_2d_nchw_fchw", text)
        self.assertIn("memref<1x3x8x8xf32>", text)
        self.assertIn("memref<4x3x3x3xf32>", text)
        self.assertIn("memref<1x4x6x6xf32>", text)

    def test_unsupported_op_raises(self):
        frontend = ManualFrontend("unsupported")
        x = frontend.add_input("x", DType.F32, (8,))
        output = frontend.hard_sigmoid(x)
        frontend.output(output)

        with self.assertRaises(MLIRBuilderError):
            MLIRBuilder().build(frontend.build())

    def test_mgpu_dialect_defines_device_ops(self):
        dialect_path = (
            Path(__file__).resolve().parents[2]
            / "compiler"
            / "mlir"
            / "mgpu_dialect.td"
        )
        text = dialect_path.read_text()

        for op_name in (
            "MGPU_LaunchOp",
            "MGPU_ThreadIdOp",
            "MGPU_BlockIdOp",
            "MGPU_GridDimOp",
            "MGPU_BlockDimOp",
            "MGPU_SharedAllocOp",
            "MGPU_BarrierOp",
            "MGPU_AsyncCopyOp",
            "MGPU_MmaOp",
        ):
            self.assertIn(op_name, text)

    def test_all_model_dtypes_have_stable_mlir_types(self):
        expected = {
            DType.F32: "f32",
            DType.F16: "f16",
            DType.BF16: "bf16",
            DType.I32: "i32",
            DType.I64: "i64",
            DType.I8: "i8",
            DType.U8: "ui8",
            DType.I4: "i4",
            DType.U4: "ui4",
            DType.F8_E4M3: "f8E4M3FN",
            DType.F8_E5M2: "f8E5M2",
            DType.BOOL: "i1",
        }

        builder = MLIRBuilder()
        for dtype, mlir_type in expected.items():
            with self.subTest(dtype=dtype):
                self.assertEqual(builder._scalar_type(dtype), mlir_type)

    def test_unknown_dtype_is_rejected(self):
        with self.assertRaises(MLIRBuilderError):
            MLIRBuilder()._scalar_type("not_a_real_dtype")


if __name__ == "__main__":
    unittest.main()
