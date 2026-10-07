// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(one-shot-bufferize)' | %filecheck %s
// CHECK: func.func @bufferize
// CHECK-NOT: tensor<
// CHECK: memref.alloc
// CHECK: linalg.fill

module {
  func.func @bufferize() {
    %0 = tensor.empty() : tensor<2x4xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4xf32>) -> tensor<2x4xf32>
    return
  }
}
