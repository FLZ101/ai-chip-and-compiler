#include "mgpu/MGPUDialect.h"
#include "mgpu/MGPUOps.h"

using namespace mlir;
using namespace mlir::mgpu;

#include "mgpu/MGPUEnums.cpp.inc"
#include "mgpu/MGPUOpsDialect.cpp.inc"

void MGPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mgpu/MGPUOps.cpp.inc"
      >();
}
