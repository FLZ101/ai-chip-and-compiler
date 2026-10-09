// RUN: %mgpu_opt %s --pass-pipeline='builtin.module(func.func(mgpu-to-isa))' | %filecheck %s
// CHECK: func.func @quant
// CHECK: "mgpu.isa"()
// CHECK-SAME: code = "MOV R0, ctaid.x
// CHECK-SAME: ROUND R5, R5
// CHECK-SAME: STG.I8 [R4], R5

module {
  func.func @quant(%arg0: memref<4xf32>) -> memref<4xi8> {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %out = memref.alloc() : memref<4xi8>
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c32, %c1, %c1) {
      "mgpu.quant"(%arg0, %out) <{scale = 5.000000e-01 : f32, zeroPoint = 0 : i32, storageType = i8}> : (memref<4xf32>, memref<4xi8>) -> ()
      mgpu.return
    }
    return %out : memref<4xi8>
  }
}
