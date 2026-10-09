// RUN: %mgpu_opt %s -o /dev/null

module {
  func.func @quant_ops(%arg0: memref<8xf32>) -> memref<8xi8> {
    %0 = memref.alloc() : memref<8xi8>
    %1 = memref.alloc() : memref<8xf32>
    %2 = memref.alloc() : memref<8xi8>
    "mgpu.quant"(%arg0, %0) <{scale = 1.000000e+00 : f32, zeroPoint = 0 : i32, storageType = i8}> : (memref<8xf32>, memref<8xi8>) -> ()
    "mgpu.dequant"(%0, %1) <{scale = 1.000000e+00 : f32, zeroPoint = 0 : i32, computeType = f32}> : (memref<8xi8>, memref<8xf32>) -> ()
    "mgpu.requant"(%0, %2) <{inputScale = 1.000000e+00 : f32, inputZeroPoint = 0 : i32, outputScale = 0.500000e+00 : f32, outputZeroPoint = 1 : i32, storageType = i8}> : (memref<8xi8>, memref<8xi8>) -> ()
    return %2 : memref<8xi8>
  }
}
