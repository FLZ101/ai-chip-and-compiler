// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(func.func(mgpu-to-isa{emit-attention-tiled}))' | %filecheck %s

// CHECK: func.func @attention_mapped
// CHECK: "mgpu.isa"()
// CHECK-SAME: MOV R0, ctaid.x
// CHECK-SAME: key_tile_loop:
// CHECK-SAME: dim_tile_loop:
// CHECK-SAME: mapped_softmax_row_loop:
// CHECK-SAME: mapped_softmax_max_loop:
// CHECK-SAME: value_v_tile_loop:
// CHECK-SAME: value_k_tile_loop:
// CHECK-SAME: STG [R37], R16
// CHECK-SAME: grid = "6x1x1"

module {
  func.func @attention_mapped(
      %q: memref<2x3x8x16xf32>, %k: memref<2x3x8x16xf32>,
      %v: memref<2x3x8x16xf32>, %mask: memref<2x3x8x8xf32>,
      %out: memref<2x3x8x16xf32>) {
    %c1 = arith.constant 1 : index
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) {
      mgpu.attention %q, %k, %v, %mask, %out {scale = 0.25 : f32, masked} : memref<2x3x8x16xf32>, memref<2x3x8x16xf32>, memref<2x3x8x16xf32>, memref<2x3x8x8xf32>, memref<2x3x8x16xf32>
      mgpu.return
    }
    return
  }
}
