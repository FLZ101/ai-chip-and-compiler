// RUN: %python %S/emit_graph.py --case rmsnorm | %filecheck %s
// RUN: %python %S/emit_graph.py --case rmsnorm | %mgpu_opt - -o /dev/null
// CHECK: func.func @rmsnorm
// CHECK: %rms_square_v0 = memref.alloc() : memref<2x4xf32>
// CHECK: math.rsqrt
// CHECK: linalg.yield %scaled : f32
