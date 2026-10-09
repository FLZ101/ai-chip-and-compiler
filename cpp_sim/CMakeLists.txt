cmake_minimum_required(VERSION 3.20)
project(cpp_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(cpp_sim_parser STATIC
  src/isa.cpp
)
target_include_directories(cpp_sim_parser PUBLIC include)

add_library(cpp_sim_memory STATIC
  src/memory.cpp
)
target_include_directories(cpp_sim_memory PUBLIC include)

add_executable(cpp_sim_parser_test tests/test_parser.cpp)
target_link_libraries(cpp_sim_parser_test PRIVATE cpp_sim_parser)

add_executable(cpp_sim_memory_test tests/test_memory.cpp)
target_link_libraries(cpp_sim_memory_test PRIVATE cpp_sim_memory)

enable_testing()
add_test(NAME cpp_sim_parser_test COMMAND cpp_sim_parser_test)
add_test(NAME cpp_sim_memory_test COMMAND cpp_sim_memory_test)
