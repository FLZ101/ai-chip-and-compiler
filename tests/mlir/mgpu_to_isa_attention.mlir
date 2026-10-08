// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(func.func(mgpu-to-isa))' | %filecheck %s
// CHECK: func.func @attention
// CHECK: "mgpu.isa"()
// CHECK-SAME: block = "32x1x1"
// CHECK-SAME: code = "IDIV R24, tid.x, 4
// CHECK-SAME: MMA R16, R0, R8, R16
// CHECK-SAME: EXP2
// CHECK-SAME: BAR.SYNC

module {
  func.func @attention(%q: memref<2x2xf32>, %k: memref<2x2xf32>, %v: memref<2x2xf32>, %mask: memref<2x2xf32>, %out: memref<2x2xf32>) {
    %c1 = arith.constant 1 : index
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) {
      mgpu.attention %q, %k, %v, %mask, %out {scale = 0.70710677 : f32, masked} : memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>
      mgpu.return
    }
    return
  }
}
