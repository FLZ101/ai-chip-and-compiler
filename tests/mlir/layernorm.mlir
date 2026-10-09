// RUN: %python %S/emit_graph.py --case layernorm | %filecheck %s
// RUN: %python %S/emit_graph.py --case layernorm | %mgpu_opt - -o /dev/null
// CHECK: func.func @layernorm
// CHECK: mgpu.layer_norm
