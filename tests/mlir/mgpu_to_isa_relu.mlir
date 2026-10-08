// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(func.func(mgpu-to-isa))' | %filecheck %s
// CHECK: func.func @relu
// CHECK: "mgpu.isa"()
// CHECK-SAME: code = "MOV R0, ctaid.x
// CHECK-SAME: MAX R6, R5, 0

module {
  func.func @relu(%arg0: memref<4xf32>) -> memref<4xf32> {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %v0 = memref.alloc() : memref<4xf32>
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c32, %c1, %c1) {
      linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%arg0 : memref<4xf32>) outs(%v0 : memref<4xf32>) {
      ^bb0(%in: f32, %out: f32):
        %act_c0 = arith.constant 0.000000e+00 : f32
        %act_value = arith.maximumf %in, %act_c0 : f32
        linalg.yield %act_value : f32
      }
      mgpu.return
    }
    return %v0 : memref<4xf32>
  }
}
