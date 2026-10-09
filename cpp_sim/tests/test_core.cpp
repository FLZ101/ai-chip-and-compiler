#include "cpp_sim/core.h"

#include <cassert>
#include <iostream>

int main() {
  cpp_sim::Device device = cpp_sim::create_device(4096, 4096, 1, 64);
  cpp_sim::LaunchConfig config;
  config.grid_dim = {1};
  config.block_dim = {8};
  cpp_sim::launch(device,
                  "MOV R0, tid.x\n"
                  "MUL R0, R0, 4\n"
                  "STG [R0], R0\n"
                  "EXIT\n",
                  config);
  for (int i = 0; i < 8; ++i)
    assert(device.global_memory.read_f32(i * 4) == static_cast<float>(i * 4));
  std::cout << "core ok\n";
  return 0;
}
