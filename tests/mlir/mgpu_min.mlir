// RUN: %mgpu_opt %s | %filecheck %s
// CHECK: module {
// CHECK:   func.func @foo() {
// CHECK:     mgpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) {
// CHECK:       %0 = mgpu.thread_id 0 : index
// CHECK:       %1 = mgpu.block_id 1 : index
// CHECK:       %2 = mgpu.grid_dim 0 : index
// CHECK:       %3 = mgpu.block_dim 2 : index
// CHECK:       mgpu.barrier
// CHECK:       mgpu.return

module {
  func.func @foo() {
    %c1 = arith.constant 1 : index
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) {
      %0 = mgpu.thread_id 0 : index
      %1 = mgpu.block_id 1 : index
      %2 = mgpu.grid_dim 0 : index
      %3 = mgpu.block_dim 2 : index
      mgpu.barrier
      mgpu.return
    }
    return
  }
}
