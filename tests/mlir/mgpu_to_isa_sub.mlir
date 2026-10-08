// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(func.func(mgpu-to-isa))' | %filecheck %s
// CHECK: func.func @sub
// CHECK: "mgpu.isa"()
// CHECK-SAME: code = "MOV R0, ctaid.x
// CHECK-SAME: SUB R6, R4, R5

module {
  func.func @sub(%arg0: memref<4xf32>, %arg1: memref<4xf32>) -> memref<4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %v0 = memref.alloc() : memref<4xf32>
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c32, %c1, %c1) {
      linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%arg0, %arg1 : memref<4xf32>, memref<4xf32>) outs(%v0 : memref<4xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %out: f32):
        %value = arith.subf %lhs, %rhs : f32
        linalg.yield %value : f32
      }
      mgpu.return
    }
    return %v0 : memref<4xf32>
  }
}
