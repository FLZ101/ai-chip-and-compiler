// RUN: %python %S/emit_graph.py --case matmul | %filecheck %s
// RUN: %python %S/emit_graph.py --case matmul | %mgpu_opt - -o /dev/null
// CHECK: module {
// CHECK:   func.func @matmul(%arg0: memref<2x4xf32>, %arg1: memref<4x3xf32>) -> memref<2x3xf32> {
// CHECK:     %v0 = memref.alloc() : memref<2x3xf32>
// CHECK:     mgpu.launch blocks(%c1, %c1, %c1) threads(%c32, %c1, %c1) {
// CHECK:       %tx_v0 = mgpu.thread_id 0 : index
// CHECK:       linalg.matmul ins(%arg0, %arg1 : memref<2x4xf32>, memref<4x3xf32>) outs(%v0 : memref<2x3xf32>)
// CHECK:       mgpu.barrier
// CHECK:       mgpu.return
// CHECK:     return %v0 : memref<2x3xf32>
