// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(func.func(mgpu-to-isa))' | %filecheck %s
// CHECK: func.func @layernorm
// CHECK: "mgpu.isa"()
// CHECK-SAME: code = "MOV R0, ctaid.x
// CHECK-SAME: RSQRT

module {
  func.func @layernorm(%x: memref<4x4xf32>, %w: memref<4xf32>, %b: memref<4xf32>, %out: memref<4x4xf32>) {
    %c1 = arith.constant 1 : index
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) {
      mgpu.layer_norm %x, %w, %b, %out : memref<4x4xf32>, memref<4xf32>, memref<4xf32>, memref<4x4xf32>
      mgpu.return
    }
    return
  }
}
