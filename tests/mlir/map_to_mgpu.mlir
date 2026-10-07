// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(func.func(map-to-mgpu))' | %filecheck %s
// CHECK: func.func @map
// CHECK: mgpu.launch blocks(%c1_0, %c1_0, %c1_0) threads(%c32, %c1_0, %c1_0)
// CHECK: mgpu.thread_id 0 : index
// CHECK: mgpu.block_id 0 : index
// CHECK: mgpu.grid_dim 0 : index
// CHECK: mgpu.block_dim 0 : index
// CHECK: scf.for
// CHECK: mgpu.return

module {
  func.func @map(%arg0: memref<8xf32>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c8 step %c1 {
      %v = arith.index_cast %i : index to i32
    }
    return
  }
}
