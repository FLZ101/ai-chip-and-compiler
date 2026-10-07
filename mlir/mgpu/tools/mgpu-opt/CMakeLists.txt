get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  MLIRMGPU
)

add_llvm_executable(mgpu-opt mgpu-opt.cpp)
llvm_update_compile_flags(mgpu-opt)
target_link_libraries(mgpu-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(mgpu-opt)
