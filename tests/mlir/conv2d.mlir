// RUN: %python %S/emit_graph.py --case conv2d | %filecheck %s
// CHECK: module {
// CHECK:   func.func @conv2d(%arg0: memref<1x3x8x8xf32>, %arg1: memref<4x3x3x3xf32>) -> memref<1x4x6x6xf32> {
// CHECK:     %v0 = memref.alloc() : memref<1x4x6x6xf32>
// CHECK:     mgpu.launch blocks(%c1, %c1, %c1) threads(%c32, %c1, %c1) {
// CHECK:       %tx_v0 = mgpu.thread_id 0 : index
// CHECK:       %v0 = linalg.conv_2d_nchw_fchw
// CHECK:         dilations = dense<1> : tensor<2xi64>
// CHECK:         strides = dense<1> : tensor<2xi64>
// CHECK:         ins(%arg0, %arg1 : memref<1x3x8x8xf32>, memref<4x3x3x3xf32>) outs(%v0 : memref<1x4x6x6xf32>) -> memref<1x4x6x6xf32>
// CHECK:       mgpu.barrier
// CHECK:       mgpu.return
// CHECK:     return %v0 : memref<1x4x6x6xf32>
