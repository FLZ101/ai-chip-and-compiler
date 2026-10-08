// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(func.func(mgpu-to-isa))' | %filecheck %s
// CHECK: func.func @conv_small
// CHECK: "mgpu.isa"()
// CHECK-SAME: block = "32x1x1"
// CHECK-SAME: code = "IDIV R24, tid.x, 4
// CHECK-SAME: MMA R16, R0, R8, R16
// CHECK-SAME: STS
// CHECK-SAME: BAR.SYNC

module {
  func.func @conv_small(%arg0: memref<1x2x4x4xf32>, %arg1: memref<2x2x2x2xf32>) -> memref<1x2x3x3xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %out = memref.alloc() : memref<1x2x3x3xf32>
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) {
      linalg.conv_2d_nchw_fchw {
        dilations = dense<1> : tensor<2xi64>,
        strides = dense<1> : tensor<2xi64>
      } ins(%arg0, %arg1 : memref<1x2x4x4xf32>, memref<2x2x2x2xf32>) outs(%out : memref<1x2x3x3xf32>)
      mgpu.return
    }
    return %out : memref<1x2x3x3xf32>
  }
}
