// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(func.func(mgpu-to-isa))' | %filecheck %s
// CHECK: func.func @sum
// CHECK: "mgpu.isa"()
// CHECK-SAME: code = "MOV R0, tid.x
// CHECK-SAME: ADD R3, R3, R5

module {
  func.func @sum(%arg0: memref<4xf32>) -> memref<f32> {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %v0 = memref.alloc() : memref<f32>
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c32, %c1, %c1) {
      linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
        iterator_types = ["reduction"]
      } ins(%arg0 : memref<4xf32>) outs(%v0 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %value = arith.addf %in, %out : f32
        linalg.yield %value : f32
      }
      mgpu.return
    }
    return %v0 : memref<f32>
  }
}
