// RUN: %python %S/emit_graph.py --case vector_add | %filecheck %s
// CHECK: module {
// CHECK:   func.func @vector_add(%arg0: memref<8xf32>, %arg1: memref<8xf32>) -> memref<8xf32> {
// CHECK:     %v0 = memref.alloc() : memref<8xf32>
// CHECK:     mgpu.launch blocks(%c1, %c1, %c1) threads(%c32, %c1, %c1) {
// CHECK:       %tx_v0 = mgpu.thread_id x
// CHECK:       %bx_v0 = mgpu.block_id x
// CHECK:       %gx_v0 = mgpu.grid_dim x
// CHECK:       %bdx_v0 = mgpu.block_dim x
// CHECK:       %v0 = linalg.generic {
// CHECK:         %value = arith.addf %lhs, %rhs : f32
// CHECK:       mgpu.barrier
// CHECK:       mgpu.return
// CHECK:     return %v0 : memref<8xf32>
