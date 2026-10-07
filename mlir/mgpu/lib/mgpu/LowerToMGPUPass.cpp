#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "mgpu/MGPUDialect.h"
#include "mgpu/MGPUOps.h"
#include "mgpu/MGPUPasses.h"

using namespace mlir;

namespace {

struct LowerToMGPUPass
    : public PassWrapper<LowerToMGPUPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToMGPUPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, mgpu::MGPUDialect>();
  }

  StringRef getArgument() const final { return "lower-to-mgpu"; }
  StringRef getDescription() const final {
    return "Insert mgpu shared memory, copies, barriers, and reserved mma ops";
  }

  void runOnOperation() override {
    SmallVector<mgpu::LaunchOp> launches;
    SmallVector<memref::SubViewOp> subviews;
    SmallVector<Operation *> matmuls;

    getOperation().walk([&](Operation *op) {
      if (auto launch = dyn_cast<mgpu::LaunchOp>(op)) {
        launches.push_back(launch);
      } else if (auto subview = dyn_cast<memref::SubViewOp>(op)) {
        subviews.push_back(subview);
      } else if (isa<linalg::MatmulOp, linalg::MatmulTransposeBOp>(op)) {
        matmuls.push_back(op);
      }
    });

    insertBarriers(launches);
    insertSharedMemoryCopies(subviews);
    insertReservedMma(matmuls);
  }

  void insertBarriers(ArrayRef<mgpu::LaunchOp> launches) {
    for (mgpu::LaunchOp launch : launches) {
      OpBuilder builder(launch);
      Block *body = &launch.getBody().front();
      builder.setInsertionPointToStart(body);
      builder.create<mgpu::BarrierOp>(launch.getLoc());
      builder.setInsertionPoint(body->getTerminator());
      builder.create<mgpu::BarrierOp>(launch.getLoc());
    }
  }

  void insertSharedMemoryCopies(ArrayRef<memref::SubViewOp> subviews) {
    for (memref::SubViewOp subview : subviews) {
      OpBuilder builder(subview);
      builder.setInsertionPointAfter(subview);
      Location loc = subview.getLoc();
      Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
      Value c128 = builder.create<arith::ConstantIndexOp>(loc, 128);
      auto sharedType = MemRefType::get({128}, builder.getI8Type());
      auto shared = builder.create<mgpu::SharedAllocOp>(loc, sharedType, c128);
      builder.create<mgpu::AsyncCopyOp>(
          loc, subview.getResult(), c0, shared, c0, c128);
    }
  }

  void insertReservedMma(ArrayRef<Operation *> matmuls) {
    for (Operation *op : matmuls) {
      auto dstOp = cast<DestinationStyleOpInterface>(op);
      if (dstOp.getDpsInputs().size() < 2 || dstOp.getDpsInits().empty())
        continue;

      OpBuilder builder(op);
      Location loc = op->getLoc();
      Value lhs = dstOp.getDpsInputs()[0];
      Value rhs = dstOp.getDpsInputs()[1];
      Value accumulator = dstOp.getDpsInits()[0];
      builder.create<mgpu::MmaOp>(
          loc, accumulator.getType(), lhs, rhs, accumulator);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::mgpu::createLowerToMGPUPass() {
  return std::make_unique<LowerToMGPUPass>();
}
