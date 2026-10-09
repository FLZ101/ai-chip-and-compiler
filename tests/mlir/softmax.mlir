// RUN: %python %S/emit_graph.py --case softmax | %filecheck %s
// RUN: %python %S/emit_graph.py --case softmax | %mgpu_opt - -o /dev/null
// CHECK: func.func @softmax
// CHECK: mgpu.softmax
