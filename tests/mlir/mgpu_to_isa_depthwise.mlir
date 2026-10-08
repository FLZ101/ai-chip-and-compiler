// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(func.func(mgpu-to-isa))' | %filecheck %s
// CHECK: func.func @depthwise
// CHECK: "mgpu.isa"()
// CHECK-SAME: code = "MOV R0, ctaid.x
// CHECK-SAME: FMA R20, R8, R11, R20

module {
  func.func @depthwise(%arg0: memref<1x2x4x4xf32>, %arg1: memref<2x2x2xf32>) -> memref<1x2x3x3xf32> {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %v0 = memref.alloc() : memref<1x2x3x3xf32>
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c32, %c1, %c1) {
      linalg.depthwise_conv_2d_nchw_chw {
        dilations = dense<1> : tensor<2xi64>,
        strides = dense<1> : tensor<2xi64>
      } ins(%arg0, %arg1 : memref<1x2x4x4xf32>, memref<2x2x2xf32>) outs(%v0 : memref<1x2x3x3xf32>)
      mgpu.return
    }
    return %v0 : memref<1x2x3x3xf32>
  }
}
