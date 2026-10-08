// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(func.func(mgpu-to-isa))' | %filecheck %s
// CHECK: func.func @maxpool
// CHECK: "mgpu.isa"()
// CHECK-SAME: code = "MOV R0, ctaid.x
// CHECK-SAME: MAX R20, R20, R8

module {
  func.func @maxpool(%arg0: memref<1x2x4x4xf32>) -> memref<1x2x2x2xf32> {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %v0 = memref.alloc() : memref<1x2x2x2xf32>
    %kernel = memref.alloc() : memref<2x2xf32>
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c32, %c1, %c1) {
      linalg.pooling_nchw_max {
        dilations = dense<1> : tensor<2xi64>,
        strides = dense<2> : tensor<2xi64>
      } ins(%arg0, %kernel : memref<1x2x4x4xf32>, memref<2x2xf32>) outs(%v0 : memref<1x2x2x2xf32>)
      mgpu.return
    }
    return %v0 : memref<1x2x2x2xf32>
  }
}
