// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(func.func(mgpu-to-isa{emit-attention-score}))' | %filecheck %s

// CHECK: func.func @attention_small
// CHECK: "mgpu.isa"()
// CHECK-SAME: MOV R41, 1
// CHECK-SAME: query_tile_loop:
// CHECK-SAME: key_tile_loop:
// CHECK-SAME: MOV R47, 4
// CHECK-SAME: dim_tile_loop:
// CHECK-SAME: MMA R16, R0, R8, R16
// CHECK-SAME: BAR.SYNC

// CHECK: func.func @attention_wide
// CHECK: "mgpu.isa"()
// CHECK-SAME: MOV R41, 2
// CHECK-SAME: query_tile_loop:
// CHECK-SAME: key_tile_loop:
// CHECK-SAME: MOV R47, 2
// CHECK-SAME: dim_tile_loop:
// CHECK-SAME: MMA R16, R0, R8, R16
// CHECK-SAME: BAR.SYNC

module {
  func.func @attention_small(
      %q: memref<1x1x8x64xf32>, %k: memref<1x1x8x64xf32>,
      %v: memref<1x1x8x64xf32>, %mask: memref<1x1x8x8xf32>,
      %out: memref<1x1x8x64xf32>) {
    %c1 = arith.constant 1 : index
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) {
      mgpu.attention %q, %k, %v, %mask, %out {scale = 0.125 : f32} : memref<1x1x8x64xf32>, memref<1x1x8x64xf32>, memref<1x1x8x64xf32>, memref<1x1x8x8xf32>, memref<1x1x8x64xf32>
      mgpu.return
    }
    return
  }

  func.func @attention_wide(
      %q: memref<1x1x32x32xf32>, %k: memref<1x1x32x32xf32>,
      %v: memref<1x1x32x32xf32>, %mask: memref<1x1x32x32xf32>,
      %out: memref<1x1x32x32xf32>) {
    %c1 = arith.constant 1 : index
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) {
      mgpu.attention %q, %k, %v, %mask, %out {scale = 0.1767766927 : f32} : memref<1x1x32x32xf32>, memref<1x1x32x32xf32>, memref<1x1x32x32xf32>, memref<1x1x32x32xf32>, memref<1x1x32x32xf32>
      mgpu.return
    }
    return
  }
}
