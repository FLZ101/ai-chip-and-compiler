// RUN: %python %S/emit_graph.py --case vector_add | %mgpu_opt - --pass-pipeline='builtin.module(func.func(mgpu-to-isa))' | %filecheck %s
// CHECK: func.func @vector_add
// CHECK: "mgpu.isa"()
// CHECK-SAME: block = "32x1x1"
// CHECK-SAME: code = "MOV R0, ctaid.x
// CHECK-SAME: grid = "1x1x1"
