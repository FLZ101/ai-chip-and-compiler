#include "mgpu/MGPUOps.h"
#include "mgpu/MGPUDialect.h"

using namespace mlir;
using namespace mlir::mgpu;

namespace {
LogicalResult verifyDimension(Operation *op, uint32_t dim) {
  if (dim > 2)
    return op->emitOpError("dimension must be 0, 1, or 2");
  return success();
}
} // namespace

#define GET_OP_CLASSES
#include "mgpu/MGPUOps.cpp.inc"

namespace mlir {
namespace mgpu {

LogicalResult LaunchOp::verify() {
  if (getBody().empty())
    return emitOpError("expected a non-empty launch body");
  if (!isa<ReturnOp>(getBody().back().getTerminator()))
    return emitOpError("launch body must end with mgpu.return");
  return success();
}

LogicalResult ThreadIdOp::verify() {
  return verifyDimension(getOperation(), getDim());
}

LogicalResult BlockIdOp::verify() {
  return verifyDimension(getOperation(), getDim());
}

LogicalResult GridDimOp::verify() {
  return verifyDimension(getOperation(), getDim());
}

LogicalResult BlockDimOp::verify() {
  return verifyDimension(getOperation(), getDim());
}

LogicalResult ReturnOp::verify() {
  Operation *parent = getOperation()->getParentOp();
  if (!parent || !isa<LaunchOp>(parent))
    return emitOpError("must be inside mgpu.launch");
  return success();
}

LogicalResult MemoryLayoutOp::verify() {
  if (getBuffers().size() != getOffsets().size())
    return emitOpError("buffer count must match offset count");
  return success();
}

} // namespace mgpu
} // namespace mlir
