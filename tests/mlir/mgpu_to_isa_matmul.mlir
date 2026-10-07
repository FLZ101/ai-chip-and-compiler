// RUN: %python %S/emit_graph.py --case matmul_square | %mgpu_opt - --pass-pipeline='builtin.module(func.func(mgpu-to-isa))' | %filecheck %s
// CHECK: func.func @matmul_square
// CHECK: "mgpu.isa"()
// CHECK-SAME: block = "4x4x1"
// CHECK-SAME: code = "MOV R0, tid.x
// CHECK-SAME: FMA R2
// CHECK-SAME: LDS
// CHECK-SAME: STS
// CHECK-SAME: BAR.SYNC
