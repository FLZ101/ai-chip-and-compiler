cmake_minimum_required(VERSION 3.20)
project(cpp_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

add_library(cpp_sim_parser STATIC
  src/isa.cpp
)
target_include_directories(cpp_sim_parser PUBLIC include)

add_library(cpp_sim_memory STATIC
  src/memory.cpp
)
target_include_directories(cpp_sim_memory PUBLIC include)

add_library(cpp_sim_core STATIC
  src/core.cpp
  src/software_kernels.cpp
)
target_include_directories(cpp_sim_core PUBLIC include)
target_link_libraries(cpp_sim_core PUBLIC cpp_sim_parser cpp_sim_memory)

add_executable(cpp_sim_parser_test tests/test_parser.cpp)
target_link_libraries(cpp_sim_parser_test PRIVATE cpp_sim_parser)

add_executable(cpp_sim_memory_test tests/test_memory.cpp)
target_link_libraries(cpp_sim_memory_test PRIVATE cpp_sim_memory)

add_executable(cpp_sim_core_test tests/test_core.cpp)
target_link_libraries(cpp_sim_core_test PRIVATE cpp_sim_core)

add_executable(cpp_sim_software_conv_test tests/test_software_conv.cpp)
target_link_libraries(cpp_sim_software_conv_test PRIVATE cpp_sim_core)

pybind11_add_module(cpp_sim MODULE src/python_bindings.cpp)
target_link_libraries(cpp_sim PRIVATE cpp_sim_core)

enable_testing()
add_test(NAME cpp_sim_parser_test COMMAND cpp_sim_parser_test)
add_test(NAME cpp_sim_memory_test COMMAND cpp_sim_memory_test)
add_test(NAME cpp_sim_core_test COMMAND cpp_sim_core_test)
add_test(NAME cpp_sim_software_conv_test COMMAND cpp_sim_software_conv_test)
