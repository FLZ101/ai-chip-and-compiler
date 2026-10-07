// RUN: %python %S/emit_graph.py --case attention | %filecheck %s
// RUN: %python %S/emit_graph.py --case attention | %mgpu_opt - -o /dev/null
// CHECK: func.func @attention
// CHECK: memref.collapse_shape
// CHECK: linalg.matmul_transpose_b
// CHECK: math.exp
// CHECK: linalg.matmul ins(%attn_weights_v0
