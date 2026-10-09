#pragma once

#include "cpp_sim/memory.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cpp_sim {

struct SoftwareKernelRequest {
  std::string op;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> weight_shape;
  std::vector<int64_t> output_shape;
  std::vector<int64_t> strides = {1, 1};
  std::vector<int64_t> paddings = {0, 0};
  std::vector<int64_t> dilations = {1, 1};
  bool has_bias = false;
  double epsilon = 1e-5;
  double pad_value = 0.0;
  int64_t input_offset = 0;
  int64_t input2_offset = 0;
  int64_t weight_offset = 0;
  int64_t bias_offset = 0;
  int64_t mean_offset = 0;
  int64_t var_offset = 0;
  int64_t output_offset = 0;
  std::vector<int64_t> kernel = {1, 1};
};

bool run_software_kernel(ByteMemory &memory, const SoftwareKernelRequest &request);

} // namespace cpp_sim
