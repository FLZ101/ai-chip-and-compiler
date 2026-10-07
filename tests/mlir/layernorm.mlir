// RUN: %python %S/emit_graph.py --case layernorm | %filecheck %s
// RUN: %python %S/emit_graph.py --case layernorm | %mgpu_opt - -o /dev/null
// CHECK: func.func @layernorm
// CHECK: %ln_sum_v0 = memref.alloc() : memref<2xf32>
// CHECK: %ln_centered_v0 = memref.alloc() : memref<2x4xf32>
// CHECK: math.rsqrt
// CHECK: linalg.yield %biased : f32
