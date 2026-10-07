#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mgpu/MGPUDialect.h"
#include "mgpu/MGPUPasses.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::mgpu::MGPUDialect>();
  mlir::mgpu::registerMGPUPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "mgpu optimizer driver\n", registry));
}
