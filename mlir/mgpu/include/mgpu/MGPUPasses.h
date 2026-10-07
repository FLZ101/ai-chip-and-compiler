#ifndef MGPU_MGPUPASSES_H
#define MGPU_MGPUPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace mgpu {

/// Pass factory functions used by ``registerMGPUPasses``.
std::unique_ptr<Pass> createTileLinalgPass();
std::unique_ptr<Pass> createMapToMGPUPass();
std::unique_ptr<Pass> createLowerToMGPUPass();

/// Register the target-specific MLIR passes used by mgpu-opt.
void registerMGPUPasses();

} // namespace mgpu
} // namespace mlir

#endif // MGPU_MGPUPASSES_H
