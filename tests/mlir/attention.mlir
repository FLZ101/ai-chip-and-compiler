// RUN: %python %S/emit_graph.py --case attention | %filecheck %s
// RUN: %python %S/emit_graph.py --case attention | %mgpu_opt - -o /dev/null
// CHECK: func.func @attention
// CHECK: mgpu.attention
