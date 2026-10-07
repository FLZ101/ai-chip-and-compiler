add_mlir_dialect_library(MLIRMGPU
  MGPUDialect.cpp
  MGPUOps.cpp
  TileLinalgPass.cpp
  MapToMGPUPass.cpp
  LowerToMGPUPass.cpp
  MGpuToIsaPass.cpp
  MGPUPasses.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mgpu

  DEPENDS
  MLIRMGPUOpsIncGen
  MLIRMGPUEnumsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSideEffectInterfaces
)
