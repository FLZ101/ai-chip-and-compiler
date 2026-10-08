// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(func.func(mgpu-to-isa))' | %filecheck %s
// CHECK: func.func @batchnorm
// CHECK: "mgpu.isa"()
// CHECK-SAME: code = "MOV R0, ctaid.x
// CHECK-SAME: RSQRT

module {
  func.func @batchnorm(%x: memref<1x2x2x2xf32>, %s: memref<2xf32>, %b: memref<2xf32>, %m: memref<2xf32>, %v: memref<2xf32>, %out: memref<1x2x2x2xf32>) {
    %c1 = arith.constant 1 : index
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) {
      mgpu.batch_norm %x, %s, %b, %m, %v, %out : memref<1x2x2x2xf32>, memref<2xf32>, memref<2xf32>, memref<2xf32>, memref<2xf32>, memref<1x2x2x2xf32>
      mgpu.return
    }
    return
  }
}
