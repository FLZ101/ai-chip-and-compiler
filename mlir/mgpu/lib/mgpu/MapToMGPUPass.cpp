#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "mgpu/MGPUDialect.h"
#include "mgpu/MGPUOps.h"
#include "mgpu/MGPUPasses.h"

using namespace mlir;

namespace {

struct MapToMGPUPass
    : public PassWrapper<MapToMGPUPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MapToMGPUPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect, mgpu::MGPUDialect>();
  }

  StringRef getArgument() const final { return "map-to-mgpu"; }
  StringRef getDescription() const final {
    return "Map top-level scf.for loops to mgpu launch/index operations";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<scf::ForOp> topLevelLoops;
    for (Operation &op : func.getBody().front()) {
      if (auto forOp = dyn_cast<scf::ForOp>(op))
        topLevelLoops.push_back(forOp);
    }

    for (scf::ForOp forOp : topLevelLoops) {
      OpBuilder builder(forOp);
      Location loc = forOp.getLoc();
      Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
      Value c32 = builder.create<arith::ConstantIndexOp>(loc, 32);
      auto launch = builder.create<mgpu::LaunchOp>(
          loc, c1, c1, c1, c32, c1, c1);

      Region &body = launch.getBody();
      if (body.empty())
        body.push_back(new Block());
      Block *entry = &body.front();

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(entry);
      auto returnOp = builder.create<mgpu::ReturnOp>(loc);
      forOp->moveBefore(returnOp);

      builder.setInsertionPointToStart(entry);
      IndexType indexType = IndexType::get(&getContext());
      builder.create<mgpu::ThreadIdOp>(loc, indexType, 0);
      builder.create<mgpu::BlockIdOp>(loc, indexType, 0);
      builder.create<mgpu::GridDimOp>(loc, indexType, 0);
      builder.create<mgpu::BlockDimOp>(loc, indexType, 0);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::mgpu::createMapToMGPUPass() {
  return std::make_unique<MapToMGPUPass>();
}
