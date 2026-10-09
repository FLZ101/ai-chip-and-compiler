#include "cpp_sim/core.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

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

} // namespace

PYBIND11_MODULE(cpp_sim, module) {
  module.doc() = "C++ simulator backend for the minimal GPU ISA";
  module.def("run", &run_global, py::arg("global_buffer"),
             py::arg("shared_size"), py::arg("sm_count"),
             py::arg("register_count"), py::arg("asm_text"),
             py::arg("grid_dim"), py::arg("block_dim"),
             py::arg("profile") = false);
}
