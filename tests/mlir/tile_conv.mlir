// RUN: %python %S/emit_graph.py --case conv2d | %mgpu_opt - --pass-pipeline='builtin.module(func.func(tile-linalg))' | %filecheck %s
// CHECK: func.func @conv2d
// CHECK: mgpu.conv2d
// CHECK-NOT: linalg.conv_2d_nchw_fchw
// CHECK-NOT: scf.for
