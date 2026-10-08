// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(func.func(mgpu-to-isa))' | %filecheck %s
// CHECK: func.func @rope
// CHECK: "mgpu.isa"()
// CHECK-SAME: code = "MOV R0, ctaid.x
// CHECK-SAME: SEL R23, R22, R18, R21

module {
  func.func @rope(%arg0: memref<2x4xf32>, %arg1: memref<2x4xf32>, %arg2: memref<2x4xf32>) -> memref<2x4xf32> {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %v0 = memref.alloc() : memref<2x4xf32>
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c32, %c1, %c1) {
      mgpu.rope %arg0, %arg1, %arg2, %v0 : memref<2x4xf32>, memref<2x4xf32>, memref<2x4xf32>, memref<2x4xf32>
      mgpu.return
    }
    return %v0 : memref<2x4xf32>
  }
}
