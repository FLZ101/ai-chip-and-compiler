#ifndef MGPU_MGPUPOS_H
#define MGPU_MGPUPOS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mgpu/MGPUEnums.h.inc"

#define GET_OP_CLASSES
#include "mgpu/MGPUOps.h.inc"

#endif // MGPU_MGPUPOS_H
