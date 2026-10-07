#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "mgpu/MGPUPasses.h"

// MLIR intentionally does not provide a general built-in ``linalg-tile`` pass.
// Tiling policy is target-specific: which ops to tile, which loops to tile,
// which tile sizes to use, and how to handle reductions/boundaries.
//
// This pass is a thin target-policy wrapper around the built-in utility
// ``mlir::linalg::tileLinalgOp``.  That utility creates the actual
// ``scf.for`` nests, ``affine.min`` boundary expressions, and
// ``memref.subview`` slices.  Keeping the policy here, rather than in Python
// or an ad hoc script, gives ``mgpu-opt`` a fixed, reusable pipeline stage:
//
//   linalg -> tile-linalg -> bufferization -> map-to-mgpu -> ...
//
using namespace mlir;

namespace {

struct TileLinalgPass
    : public PassWrapper<TileLinalgPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileLinalgPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    memref::MemRefDialect>();
  }

  StringRef getArgument() const final { return "tile-linalg"; }
  StringRef getDescription() const final {
    return "Tile linalg operations for the mgpu target";
  }

  void runOnOperation() override {
    SmallVector<linalg::LinalgOp> linalgOps;
    getOperation().walk([&](linalg::LinalgOp op) {
      // Phase 3.2 currently tiles buffer-form linalg operations. Tensor-form
      // tiling is handled by the same infrastructure after bufferization.
      if (op->getNumResults() == 0)
        linalgOps.push_back(op);
    });

    IRRewriter rewriter(&getContext());
    for (linalg::LinalgOp op : linalgOps) {
      SmallVector<int64_t> tileSizes = chooseTileSizes(op);
      linalg::LinalgTilingOptions options;
      options.setTileSizes(tileSizes);
      options.setLoopType(linalg::LinalgTilingLoopType::Loops);

      if (failed(linalg::tileLinalgOp(rewriter, op, options))) {
        op.emitWarning("tile-linalg: unable to tile operation");
        continue;
      }
      rewriter.eraseOp(op);
    }
  }

  static SmallVector<int64_t> chooseTileSizes(linalg::LinalgOp op) {
    if (isa<linalg::MatmulOp, linalg::MatmulTransposeBOp,
            linalg::BatchMatmulOp>(op.getOperation())) {
      return {16, 16, 16};
    }
    if (isa<linalg::Conv2DNchwFchwOp, linalg::PoolingNchwMaxOp,
            linalg::PoolingNchwSumOp>(op.getOperation())) {
      return {1, 1, 8, 8};
    }
    if (op.getNumLoops() <= 2) {
      return {16, 16};
    }
    return {16, 16, 16};
  }
};

} // namespace

void mlir::mgpu::registerMGPUPasses() {
  PassRegistration<TileLinalgPass>();
}
