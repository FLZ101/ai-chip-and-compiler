// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(func.func(mgpu-to-isa{emit-attention-tiled}))' | %filecheck %s

// CHECK: func.func @attention_full
// CHECK: "mgpu.isa"()
// CHECK-SAME: key_tile_loop:
// CHECK-SAME: dim_tile_loop:
// CHECK-SAME: MMA R16, R0, R8, R16
// CHECK-SAME: mapped_softmax_row_loop:
// CHECK-SAME: mapped_softmax_max_loop:
// CHECK-SAME: mapped_softmax_sum_loop:
// CHECK-SAME: mapped_softmax_store_loop:
// CHECK-SAME: EXP2
// CHECK-SAME: value_v_tile_loop:
// CHECK-SAME: value_k_tile_loop:
// CHECK-SAME: MMA R16, R0, R8, R16
// CHECK-SAME: STG [R37], R16
// CHECK-SAME: EXIT

module {
  func.func @attention_full(
      %q: memref<1x1x8x64xf32>, %k: memref<1x1x8x64xf32>,
      %v: memref<1x1x8x64xf32>, %mask: memref<1x1x8x8xf32>,
      %out: memref<1x1x8x64xf32>) {
    %c1 = arith.constant 1 : index
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) {
      mgpu.attention %q, %k, %v, %mask, %out {scale = 0.125 : f32, masked} : memref<1x1x8x64xf32>, memref<1x1x8x64xf32>, memref<1x1x8x64xf32>, memref<1x1x8x8xf32>, memref<1x1x8x64xf32>
      mgpu.return
    }
    return
  }
}
