// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(func.func(mgpu-to-isa))' | %filecheck %s
// CHECK: func.func @spill
// CHECK: "mgpu.isa"()
// CHECK-SAME: code = "// mgpu.launch generic lowering\0AMOV R0, tid.x\0AMOV R1, tid.x\0AMOV R2, tid.x\0AMOV R3, tid.x\0ASTG [4096 + 0], R0\0AMOV R0, tid.x\0ASTG [4096 + 4], R1\0AMOV R1, tid.x\0AEXIT\0A"

module {
  func.func @spill() {
    %c1 = arith.constant 1 : index
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) {
      %t0 = mgpu.thread_id 0 : index
      %t1 = mgpu.thread_id 0 : index
      %t2 = mgpu.thread_id 0 : index
      %t3 = mgpu.thread_id 0 : index
      %t4 = mgpu.thread_id 0 : index
      %t5 = mgpu.thread_id 0 : index
      mgpu.return
    }
    return
  }
}
