// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(func.func(mgpu-to-isa{emit-attention-softmax}))' | %filecheck %s

// CHECK: func.func @attention_softmax
// CHECK: "mgpu.isa"()
// CHECK-SAME: query_tile_loop:
// CHECK-SAME: key_tile_loop:
// CHECK-SAME: dim_tile_loop:
// CHECK-SAME: MMA R16, R0, R8, R16
// CHECK-SAME: BAR.SYNC
// CHECK-SAME: softmax_row_loop:
// CHECK-SAME: softmax_max_loop:
// CHECK-SAME: softmax_sum_loop:
// CHECK-SAME: EXP2
// CHECK-SAME: softmax_store_loop:
// CHECK-SAME: BAR.SYNC

module {
  func.func @attention_softmax(
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
