// RUN: %mgpu_opt %s -o - | %filecheck %s
// CHECK: func.func @layout
// CHECK: mgpu.memory_layout
// CHECK-SAME: offsets = [0, 16]

module {
  func.func @layout(%a: memref<4xf32>, %b: memref<4xf32>) {
    mgpu.memory_layout %a, %b offsets = [0, 16] : memref<4xf32>, memref<4xf32>
    return
  }
}
