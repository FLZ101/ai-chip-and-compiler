// RUN: %python %S/emit_graph.py --case softmax | %mgpu_opt - --pass-pipeline='builtin.module(func.func(tile-linalg))' | %filecheck %s
// CHECK: func.func @softmax
// CHECK: scf.for
// CHECK: memref.subview
// CHECK: math.exp
