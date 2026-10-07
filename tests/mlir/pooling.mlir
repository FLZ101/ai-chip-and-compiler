// RUN: %python %S/emit_graph.py --case pooling | %filecheck %s
// RUN: %python %S/emit_graph.py --case pooling | %mgpu_opt - -o /dev/null
// CHECK: func.func @pooling
// CHECK: %pool_kernel_v0 = memref.alloc() : memref<2x2xf32>
// CHECK: linalg.pooling_nchw_max
