// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(func.func(lower-to-mgpu))' | %filecheck %s
// CHECK: func.func @lower
// CHECK: mgpu.barrier
// CHECK: "mgpu.shared_alloc"
// CHECK: "mgpu.async_copy"
// CHECK: "mgpu.mma"
// CHECK: mgpu.barrier

module {
  func.func @lower(%A: memref<8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) {
      %0 = memref.subview %A[%c0, %c0] [%c4, %c4] [%c1, %c1] : memref<8x8xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      linalg.matmul ins(%0, %B : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<8x8xf32>) outs(%C : memref<8x8xf32>)
      mgpu.return
    }
    return
  }
}
