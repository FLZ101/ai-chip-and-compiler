#include "cpp_sim/core.h"
#include "cpp_sim/software_kernels.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

std::vector<int64_t> to_i64_vector(const py::tuple &value) {
  std::vector<int64_t> result;
  for (auto item : value)
    result.push_back(item.cast<int64_t>());
  return result;
}

py::dict run_global(py::buffer global_buffer, int shared_size, int sm_count,
                    int register_count, const std::string &asm_text,
                    const py::tuple &grid_dim_tuple,
                    const py::tuple &block_dim_tuple, bool profile_flag) {
  py::buffer_info info = global_buffer.request();
  if (info.format != py::format_descriptor<uint8_t>::format())
    throw std::runtime_error("global buffer must be a byte buffer");
  if (info.ndim != 1)
    throw std::runtime_error("global buffer must be one-dimensional");

  const auto *src = static_cast<const uint8_t *>(info.ptr);
  std::vector<uint8_t> bytes(src, src + static_cast<size_t>(info.size));

  cpp_sim::Device device =
      cpp_sim::create_device(bytes.size(), shared_size, sm_count, register_count);
  std::memcpy(device.global_memory.bytes().data(), bytes.data(), bytes.size());

  std::vector<int> grid_dim;
  std::vector<int> block_dim;
  for (auto item : grid_dim_tuple)
    grid_dim.push_back(item.cast<int>());
  for (auto item : block_dim_tuple)
    block_dim.push_back(item.cast<int>());

  cpp_sim::LaunchConfig config;
  config.grid_dim = std::move(grid_dim);
  config.block_dim = std::move(block_dim);
  config.profile = profile_flag;
  cpp_sim::ProfileCounters profile = cpp_sim::launch(device, asm_text, config);

  std::memcpy(info.ptr, device.global_memory.bytes().data(), bytes.size());

  py::dict result;
  result["cycles"] = profile.cycles;
  result["issued_instructions"] = profile.issued_instructions;
  result["global_load_bytes"] = profile.global_load_bytes;
  result["global_store_bytes"] = profile.global_store_bytes;
  result["shared_load_bytes"] = profile.shared_load_bytes;
  result["shared_store_bytes"] = profile.shared_store_bytes;
  result["shared_bank_conflicts"] = profile.shared_bank_conflicts;
  return result;
}

bool software_run(py::buffer global_buffer, const py::dict &request_dict) {
  py::buffer_info info = global_buffer.request();
  if (info.format != py::format_descriptor<uint8_t>::format())
    throw std::runtime_error("global buffer must be a byte buffer");
  const auto *src = static_cast<const uint8_t *>(info.ptr);
  std::vector<uint8_t> bytes(src, src + static_cast<size_t>(info.size));
  cpp_sim::ByteMemory memory(bytes.size(), "software");
  std::memcpy(memory.bytes().data(), bytes.data(), bytes.size());

  cpp_sim::SoftwareKernelRequest request;
  request.op = request_dict["op"].cast<std::string>();
  request.input_shape = to_i64_vector(request_dict["input_shape"].cast<py::tuple>());
  request.weight_shape = request_dict.contains("weight_shape")
                             ? to_i64_vector(request_dict["weight_shape"].cast<py::tuple>())
                             : std::vector<int64_t>{};
  request.output_shape = to_i64_vector(request_dict["output_shape"].cast<py::tuple>());
  request.strides = request_dict.contains("strides")
                        ? to_i64_vector(request_dict["strides"].cast<py::tuple>())
                        : std::vector<int64_t>{1, 1};
  request.paddings = request_dict.contains("paddings")
                         ? to_i64_vector(request_dict["paddings"].cast<py::tuple>())
                         : std::vector<int64_t>{0, 0};
  request.dilations = request_dict.contains("dilations")
                          ? to_i64_vector(request_dict["dilations"].cast<py::tuple>())
                          : std::vector<int64_t>{1, 1};
  request.has_bias = request_dict.contains("has_bias")
                         ? request_dict["has_bias"].cast<bool>()
                         : false;
  request.epsilon = request_dict.contains("epsilon")
                        ? request_dict["epsilon"].cast<double>()
                        : 1e-5;
  request.input_offset = request_dict["input_offset"].cast<int64_t>();
  request.input2_offset = request_dict.contains("input2_offset")
                              ? request_dict["input2_offset"].cast<int64_t>()
                              : 0;
  request.weight_offset = request_dict.contains("weight_offset")
                              ? request_dict["weight_offset"].cast<int64_t>()
                              : 0;
  request.bias_offset = request_dict.contains("bias_offset")
                            ? request_dict["bias_offset"].cast<int64_t>()
                            : 0;
  request.mean_offset = request_dict.contains("mean_offset")
                            ? request_dict["mean_offset"].cast<int64_t>()
                            : 0;
  request.var_offset = request_dict.contains("var_offset")
                           ? request_dict["var_offset"].cast<int64_t>()
                           : 0;
  request.output_offset = request_dict["output_offset"].cast<int64_t>();
  request.kernel = to_i64_vector(
      request_dict.contains("kernel")
          ? request_dict["kernel"].cast<py::tuple>()
          : py::make_tuple(1, 1));

  bool ok = cpp_sim::run_software_kernel(memory, request);
  std::memcpy(info.ptr, memory.bytes().data(), bytes.size());
  return ok;
}

} // namespace

PYBIND11_MODULE(cpp_sim, module) {
  module.doc() = "C++ simulator backend for the minimal GPU ISA";
  module.def("run", &run_global, py::arg("global_buffer"),
             py::arg("shared_size"), py::arg("sm_count"),
             py::arg("register_count"), py::arg("asm_text"),
             py::arg("grid_dim"), py::arg("block_dim"),
             py::arg("profile") = false);
  module.def("software_run", &software_run, py::arg("global_buffer"),
             py::arg("request"));
}
