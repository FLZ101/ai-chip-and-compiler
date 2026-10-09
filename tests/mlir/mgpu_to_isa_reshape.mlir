// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(func.func(mgpu-to-isa))' | %filecheck %s
// CHECK: func.func @reshape
// CHECK: "mgpu.isa"()
// CHECK-SAME: code = "MOV R0, ctaid.x
// CHECK-SAME: ADD R4, R4, 64

module {
  func.func @reshape(%arg0: memref<16xf32>) -> memref<4x4xf32> {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %v0 = memref.alloc() : memref<4x4xf32>
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c32, %c1, %c1) {
      mgpu.reshape %arg0, %v0 : memref<16xf32>, memref<4x4xf32>
      mgpu.return
    }
    return %v0 : memref<4x4xf32>
  }
}
