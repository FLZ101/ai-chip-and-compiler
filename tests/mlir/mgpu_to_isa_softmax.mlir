// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(func.func(mgpu-to-isa))' | %filecheck %s
// CHECK: func.func @softmax
// CHECK: "mgpu.isa"()
// CHECK-SAME: code = "MOV R0, ctaid.x
// CHECK-SAME: EXP2

module {
  func.func @softmax(%x: memref<2x4xf32>, %out: memref<2x4xf32>) {
    %c1 = arith.constant 1 : index
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) {
      mgpu.softmax %x, %out : memref<2x4xf32>, memref<2x4xf32>
      mgpu.return
    }
    return
  }
}
