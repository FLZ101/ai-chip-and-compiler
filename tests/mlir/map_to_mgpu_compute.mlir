// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(func.func(map-to-mgpu))' | %filecheck %s
// CHECK: func.func @map_compute
// CHECK: mgpu.launch
// CHECK: linalg.generic
// CHECK: mgpu.launch
// CHECK: linalg.pooling_nchw_max

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
  func.func @map_compute(%arg0: memref<1x2x4x4xf32>) -> memref<1x2x2x2xf32> {
    %kernel = memref.alloc() : memref<2x2xf32>
    %out = memref.alloc() : memref<1x2x2x2xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<1x2x4x4xf32>) outs(%arg0 : memref<1x2x4x4xf32>) {
    ^bb0(%in: f32, %out_arg: f32):
      linalg.yield %in : f32
    }
    linalg.pooling_nchw_max {
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<2> : tensor<2xi64>
    } ins(%arg0, %kernel : memref<1x2x4x4xf32>, memref<2x2xf32>) outs(%out : memref<1x2x2x2xf32>)
    return %out : memref<1x2x2x2xf32>
  }
}
