add_mlir_dialect(MGPUOps mgpu)
add_mlir_doc(MGPUDialect MGPUDialect mgpu/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS MGPUOps.td)
mlir_tablegen(MGPUEnums.h.inc -gen-enum-decls)
mlir_tablegen(MGPUEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRMGPUEnumsIncGen)
