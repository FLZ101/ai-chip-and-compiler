// RUN: %python %S/emit_graph.py --case rmsnorm | %filecheck %s
// RUN: %python %S/emit_graph.py --case rmsnorm | %mgpu_opt - -o /dev/null
// CHECK: func.func @rmsnorm
// CHECK: mgpu.rms_norm
