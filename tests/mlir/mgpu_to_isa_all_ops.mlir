// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(func.func(mgpu-to-isa))' | %filecheck %s
// CHECK: func.func @all_ops
// CHECK: "mgpu.isa"()
// CHECK-SAME: code = "// mgpu.launch generic lowering\0AMOV R0, tid.x\0AMOV R1, ctaid.x\0AMOV R2, nctaid.x\0AMOV R3, ntid.x\0A// mgpu.shared_alloc\0A// mgpu.async_copy\0A// mgpu.quant\0A// mgpu.dequant\0A// mgpu.requant\0A// mgpu.mma reserved\0ABAR.SYNC\0AEXIT\0A"

module {
  func.func @all_ops(%arg0: memref<8xf32>) {
    %c1 = arith.constant 1 : index
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) {
      %tx = mgpu.thread_id 0 : index
      %bx = mgpu.block_id 0 : index
      %gx = mgpu.grid_dim 0 : index
      %bd = mgpu.block_dim 0 : index
      %shared = "mgpu.shared_alloc"(%c1) : (index) -> memref<128xi8>
      %q = memref.alloc() : memref<8xi8>
      %dq = memref.alloc() : memref<8xf32>
      %rq = memref.alloc() : memref<8xi8>
      "mgpu.async_copy"(%arg0, %c1, %shared, %c1, %c1) : (memref<8xf32>, index, memref<128xi8>, index, index) -> ()
      "mgpu.quant"(%arg0, %q) <{scale = 1.000000e+00 : f32, zeroPoint = 0 : i32, storageType = i8}> : (memref<8xf32>, memref<8xi8>) -> ()
      "mgpu.dequant"(%q, %dq) <{scale = 1.000000e+00 : f32, zeroPoint = 0 : i32, computeType = f32}> : (memref<8xi8>, memref<8xf32>) -> ()
      "mgpu.requant"(%q, %rq) <{inputScale = 1.000000e+00 : f32, inputZeroPoint = 0 : i32, outputScale = 0.500000e+00 : f32, outputZeroPoint = 1 : i32, storageType = i8}> : (memref<8xi8>, memref<8xi8>) -> ()
      %mma = "mgpu.mma"(%arg0, %arg0, %arg0) : (memref<8xf32>, memref<8xf32>, memref<8xf32>) -> memref<8xf32>
      mgpu.barrier
      mgpu.return
    }
    return
  }
}
