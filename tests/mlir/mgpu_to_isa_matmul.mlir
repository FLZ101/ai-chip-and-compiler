// RUN: %python %S/emit_graph.py --case matmul_square | %mgpu_opt - --pass-pipeline='builtin.module(func.func(mgpu-to-isa))' | %filecheck %s
// CHECK: func.func @matmul_square
// CHECK: "mgpu.isa"()
// CHECK-SAME: block = "32x1x1"
// CHECK-SAME: code = "IDIV R24, tid.x, 4
// CHECK-SAME: MMA R16, R0, R8, R16
