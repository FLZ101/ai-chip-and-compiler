// RUN: %python %S/emit_graph.py --case softmax | %filecheck %s
// RUN: %python %S/emit_graph.py --case softmax | %mgpu_opt - -o /dev/null
// CHECK: func.func @softmax
// CHECK: %softmax_max_v0 = memref.alloc() : memref<2xf32>
// CHECK: math.exp
// CHECK: linalg.yield %div : f32
