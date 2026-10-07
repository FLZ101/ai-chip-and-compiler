// RUN: %python %S/emit_graph.py --case vector_add | %filecheck %s
// RUN: %python %S/emit_graph.py --case vector_add | %mgpu_opt - -o /dev/null
// CHECK: module {
// CHECK:   func.func @vector_add(%arg0: memref<8xf32>, %arg1: memref<8xf32>) -> memref<8xf32> {
// CHECK:     %v0 = memref.alloc() : memref<8xf32>
// CHECK:     mgpu.launch blocks(%c1, %c1, %c1) threads(%c32, %c1, %c1) {
// CHECK:       %tx_v0 = mgpu.thread_id 0 : index
// CHECK:       %bx_v0 = mgpu.block_id 0 : index
// CHECK:       %gx_v0 = mgpu.grid_dim 0 : index
// CHECK:       %bdx_v0 = mgpu.block_dim 0 : index
// CHECK:       linalg.generic {
// CHECK:         %value = arith.addf %lhs, %rhs : f32
// CHECK:       mgpu.barrier
// CHECK:       mgpu.return
// CHECK:     return %v0 : memref<8xf32>
