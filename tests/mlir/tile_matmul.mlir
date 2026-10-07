// RUN: %python %S/emit_graph.py --case matmul | %mgpu_opt - --pass-pipeline='builtin.module(func.func(tile-linalg))' | %filecheck %s
// CHECK: func.func @matmul
// CHECK: scf.for
// CHECK: memref.subview
// CHECK: linalg.matmul
