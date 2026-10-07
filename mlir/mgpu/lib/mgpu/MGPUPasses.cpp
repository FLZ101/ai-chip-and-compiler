#include "mlir/Pass/PassRegistry.h"

#include "mgpu/MGPUPasses.h"

void mlir::mgpu::registerMGPUPasses() {
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createTileLinalgPass(); });
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createMapToMGPUPass(); });
}
