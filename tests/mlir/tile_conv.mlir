// RUN: %python %S/emit_graph.py --case conv2d | %mgpu_opt - --pass-pipeline='builtin.module(func.func(tile-linalg))' | %filecheck %s
// CHECK: func.func @conv2d
// CHECK: scf.for
// CHECK: memref.subview
// CHECK: linalg.conv_2d_nchw_fchw
