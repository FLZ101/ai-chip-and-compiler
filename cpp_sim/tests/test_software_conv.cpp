#include "cpp_sim/software_kernels.h"

#include <cassert>
#include <iostream>

int main() {
  cpp_sim::ByteMemory memory(4096, "test");
  memory.write_f32(0, 1.0f);
  memory.write_f32(4, 2.0f);
  memory.write_f32(8, 3.0f);
  memory.write_f32(12, 4.0f);
  memory.write_f32(64, 1.0f);
  memory.write_f32(68, 1.0f);
  memory.write_f32(72, 1.0f);
  memory.write_f32(76, 1.0f);

  cpp_sim::SoftwareKernelRequest request;
  request.op = "conv2d";
  request.input_shape = {1, 1, 2, 2};
  request.weight_shape = {1, 1, 2, 2};
  request.output_shape = {1, 1, 1, 1};
  request.strides = {1, 1};
  request.paddings = {0, 0};
  request.dilations = {1, 1};
  request.input_offset = 0;
  request.weight_offset = 64;
  request.output_offset = 128;

  assert(cpp_sim::run_software_kernel(memory, request));
  assert(memory.read_f32(128) == 10.0f);
  std::cout << "software conv ok\n";
  return 0;
}
