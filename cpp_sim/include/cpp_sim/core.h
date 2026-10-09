#pragma once

#include "cpp_sim/isa.h"
#include "cpp_sim/memory.h"

#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace cpp_sim {

struct StallCounters {
  int64_t fetch = 0;
  int64_t alu = 0;
  int64_t memory = 0;
  int64_t barrier = 0;
  int64_t scoreboard = 0;
};

struct ProfileCounters {
  int64_t cycles = 0;
  int64_t issued_instructions = 0;
  StallCounters stalls;
  int64_t global_load_bytes = 0;
  int64_t global_store_bytes = 0;
  int64_t shared_load_bytes = 0;
  int64_t shared_store_bytes = 0;
  int64_t shared_bank_conflicts = 0;
};

struct ReconvergenceState {
  int pending_pc = 0;
  uint32_t pending_mask = 0;
  int reconvergence_pc = 0;
  uint32_t original_mask = 0;
  std::string phase = "taken";
};

struct LaneState {
  int lane_id = 0;
  std::vector<double> regs;
  bool active = true;
  std::unordered_map<std::string, int> reg_ready_cycle;
};

struct WarpState {
  int warp_id = 0;
  std::vector<LaneState> lanes;
  int pc = 0;
  bool exited = false;
  uint32_t active_mask = 0xffffffffu;
  std::vector<ReconvergenceState> reconvergence_stack;
  bool waiting_at_barrier = false;
};

struct SM {
  int sm_id = 0;
  ByteMemory shared_memory{0, "shared"};
  std::vector<WarpState> warps;
  int next_warp_index = 0;
  int block_linear_id = 0;
  std::vector<int> block_idx = {0, 0, 0};
};

struct Device {
  ByteMemory global_memory{0, "global"};
  std::vector<SM> sms;
  int register_count = 64;
  std::vector<int> grid_dim = {1};
  std::vector<int> block_dim = {1};
  int64_t clock = 0;
  ProfileCounters profile;
};

struct LaunchConfig {
  std::vector<int> grid_dim = {1};
  std::vector<int> block_dim = {1};
  bool profile = false;
};

Device create_device(size_t global_size, size_t shared_size, int sm_count,
                     int register_count);
ProfileCounters launch(Device &device, const std::string &asm_text,
                       const LaunchConfig &config);

std::vector<int> unflatten_index(int index, const std::vector<int> &shape);

} // namespace cpp_sim
