// RUN: %python %S/emit_graph.py --case attention | %mgpu_opt - --pass-pipeline='builtin.module(func.func(tile-linalg))' | %filecheck %s
// CHECK: func.func @attention
// CHECK: mgpu.attention
