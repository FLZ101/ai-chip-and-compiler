#include "cpp_sim/core.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace cpp_sim {

namespace {

bool is_global_memory_op(const std::string &op) {
  return op.rfind("LDG", 0) == 0 || op.rfind("STG", 0) == 0;
}

bool is_shared_memory_op(const std::string &op) {
  return op.rfind("LDS", 0) == 0 || op.rfind("STS", 0) == 0;
}

std::string memory_type(const std::string &op) {
  size_t dot = op.find('.');
  if (dot == std::string::npos)
    return "f32";
  return op.substr(dot + 1);
}

double read_typed(ByteMemory &memory, const std::string &type, int64_t addr) {
  if (type == "f32")
    return memory.read_f32(addr);
  if (type == "F16")
    return memory.read_f16(addr);
  if (type == "BF16")
    return memory.read_bf16(addr);
  if (type == "F8E4M3")
    return memory.read_f8_e4m3(addr);
  if (type == "F8E5M2")
    return memory.read_f8_e5m2(addr);
  if (type == "I8")
    return memory.read_i8(addr);
  if (type == "U8")
    return memory.read_u8(addr);
  if (type == "I4")
    return memory.read_i4(addr);
  if (type == "U4")
    return memory.read_u4(addr);
  if (type == "I32")
    return memory.read_i32(addr);
  throw std::runtime_error("unsupported memory type: " + type);
}

void write_typed(ByteMemory &memory, const std::string &type, int64_t addr,
                 double value) {
  if (type == "f32")
    memory.write_f32(addr, static_cast<float>(value));
  else if (type == "F16")
    memory.write_f16(addr, static_cast<float>(value));
  else if (type == "BF16")
    memory.write_bf16(addr, static_cast<float>(value));
  else if (type == "F8E4M3")
    memory.write_f8_e4m3(addr, static_cast<float>(value));
  else if (type == "F8E5M2")
    memory.write_f8_e5m2(addr, static_cast<float>(value));
  else if (type == "I8")
    memory.write_i8(addr, static_cast<int>(value));
  else if (type == "U8")
    memory.write_u8(addr, static_cast<uint8_t>(value));
  else if (type == "I4")
    memory.write_i4(addr, static_cast<int>(value));
  else if (type == "U4")
    memory.write_u4(addr, static_cast<uint8_t>(value));
  else if (type == "I32")
    memory.write_i32(addr, static_cast<int32_t>(value));
  else
    throw std::runtime_error("unsupported memory type: " + type);
}

double special_register_value(Device &device, SM &sm, WarpState &warp,
                              LaneState &lane, const std::string &name) {
  if (name == "tid.x")
    return lane.lane_id % device.block_dim[0];
  if (name == "tid.y")
    return (lane.lane_id / device.block_dim[0]) % device.block_dim[1];
  if (name == "tid.z")
    return lane.lane_id /
           (device.block_dim[0] * device.block_dim[1]);
  if (name == "ntid.x")
    return device.block_dim[0];
  if (name == "ntid.y")
    return device.block_dim[1];
  if (name == "ntid.z")
    return device.block_dim[2];
  if (name == "ctaid.x")
    return sm.block_idx[0];
  if (name == "ctaid.y")
    return sm.block_idx[1];
  if (name == "ctaid.z")
    return sm.block_idx[2];
  if (name == "nctaid.x")
    return device.grid_dim[0];
  if (name == "nctaid.y")
    return device.grid_dim[1];
  if (name == "nctaid.z")
    return device.grid_dim[2];
  if (name == "laneid")
    return lane.lane_id;
  if (name == "warpid")
    return warp.warp_id;
  if (name == "clock")
    return static_cast<double>(device.clock);
  throw std::runtime_error("unknown special register: " + name);
}

double read_operand(Device &device, SM &sm, WarpState &warp, LaneState &lane,
                    const Operand &operand) {
  if (operand.kind == OperandKind::Reg) {
    if (operand.special)
      return special_register_value(device, sm, warp, lane, operand.text);
    if (operand.reg_index < 0 ||
        operand.reg_index >= static_cast<int>(lane.regs.size()))
      throw std::runtime_error("unknown register: " + operand.text);
    return lane.regs[operand.reg_index];
  }
  if (operand.kind == OperandKind::Imm)
    return operand.imm;
  if (operand.kind == OperandKind::Mem) {
    double base = 0;
    if (operand.base_special)
      base = special_register_value(device, sm, warp, lane, operand.base);
    else {
      if (operand.base_reg_index < 0 ||
          operand.base_reg_index >= static_cast<int>(lane.regs.size()))
        throw std::runtime_error("unknown memory base: " + operand.base);
      base = lane.regs[operand.base_reg_index];
    }
    return base + operand.offset;
  }
  throw std::runtime_error("unsupported operand");
}

void write_register(LaneState &lane, const Operand &dst, double value) {
  if (dst.kind != OperandKind::Reg)
    throw std::runtime_error("destination is not a register");
  if (dst.reg_index < 0 || dst.reg_index >= static_cast<int>(lane.regs.size()))
    throw std::runtime_error("unknown destination register");
  lane.regs[dst.reg_index] = value;
}

double alu(const std::string &op, double a, double b, double c) {
  if (op == "MOV")
    return a;
  if (op == "NEG")
    return -a;
  if (op == "ABS")
    return std::abs(a);
  if (op == "ROUND")
    return std::round(a);
  if (op == "I2F")
    return a;
  if (op == "RCP")
    return 1.0 / a;
  if (op == "RSQRT")
    return 1.0 / std::sqrt(a);
  if (op == "EXP2")
    return std::exp2(a);
  if (op == "LOG2")
    return std::log2(a);
  if (op == "FMA")
    return a * b + c;
  if (op == "ADD")
    return a + b;
  if (op == "SUB")
    return a - b;
  if (op == "MUL")
    return a * b;
  if (op == "DIV")
    return a / b;
  if (op == "IDIV")
    return static_cast<double>(static_cast<int64_t>(a) /
                               static_cast<int64_t>(b));
  if (op == "MAX")
    return std::max(a, b);
  if (op == "MIN")
    return std::min(a, b);
  throw std::runtime_error("unsupported ALU opcode: " + op);
}

bool comparison(const std::string &op, double a, double b) {
  if (op == "CMP" || op == "CMP.EQ")
    return a == b;
  if (op == "CMP.NE")
    return a != b;
  if (op == "CMP.LT")
    return a < b;
  if (op == "CMP.LE")
    return a <= b;
  if (op == "CMP.GT")
    return a > b;
  if (op == "CMP.GE")
    return a >= b;
  throw std::runtime_error("unsupported compare: " + op);
}

std::vector<LaneState *> active_lanes(WarpState &warp) {
  std::vector<LaneState *> lanes;
  for (auto &lane : warp.lanes) {
    if (lane.active && (warp.active_mask & (1u << lane.lane_id)))
      lanes.push_back(&lane);
  }
  return lanes;
}

bool apply_reconvergence(WarpState &warp) {
  if (warp.reconvergence_stack.empty())
    return false;
  ReconvergenceState &state = warp.reconvergence_stack.back();
  if (warp.pc != state.reconvergence_pc)
    return false;
  if (state.phase == "taken") {
    warp.active_mask = state.pending_mask;
    warp.pc = state.pending_pc;
    state.phase = "fallthrough";
    return true;
  }
  warp.active_mask = state.original_mask;
  warp.reconvergence_stack.pop_back();
  return true;
}

void execute_instruction(Device &device, SM &sm, WarpState &warp,
                         const Program &program) {
  while (apply_reconvergence(warp)) {
  }
  if (warp.pc < 0 || warp.pc >= static_cast<int>(program.instructions.size())) {
    warp.exited = true;
    return;
  }
  const Instruction &inst = program.instructions[warp.pc];

  if (inst.op == "EXIT" || inst.op == "RET") {
    warp.exited = true;
    return;
  }

  if (inst.op == "BAR.SYNC") {
    warp.waiting_at_barrier = true;
    bool all_waiting = true;
    for (const auto &other : sm.warps) {
      if (!other.waiting_at_barrier) {
        all_waiting = false;
        break;
      }
    }
    if (all_waiting) {
      for (auto &other : sm.warps) {
        other.waiting_at_barrier = false;
        other.pc += 1;
      }
    }
    return;
  }

  if (inst.op == "BRA") {
    if (inst.operands.empty() ||
        inst.operands[0].kind != OperandKind::Label)
      throw std::runtime_error("BRA requires a label target");
    int target = program.labels.at(inst.operands[0].text);
    if (inst.operands.size() == 1) {
      warp.pc = target;
      return;
    }
    uint32_t taken_mask = 0;
    for (LaneState *lane : active_lanes(warp)) {
      if (read_operand(device, sm, warp, *lane, inst.operands[1]) != 0)
        taken_mask |= 1u << lane->lane_id;
    }
    uint32_t fallthrough_mask = warp.active_mask & ~taken_mask;
    if (taken_mask == 0) {
      warp.pc += 1;
      return;
    }
    if (fallthrough_mask == 0) {
      warp.active_mask = taken_mask;
      warp.pc = target;
      return;
    }
    if (inst.operands.size() != 3 ||
        inst.operands[2].kind != OperandKind::Label)
      throw std::runtime_error("divergent BRA requires a reconvergence label");
    ReconvergenceState state;
    state.pending_pc = warp.pc + 1;
    state.pending_mask = fallthrough_mask;
    state.reconvergence_pc = program.labels.at(inst.operands[2].text);
    state.original_mask = warp.active_mask;
    warp.reconvergence_stack.push_back(state);
    warp.active_mask = taken_mask;
    warp.pc = target;
    return;
  }

  bool memory_store = is_shared_memory_op(inst.op) || is_global_memory_op(inst.op);
  bool memory_load = memory_store && inst.op.rfind("LD", 0) == 0;
  bool compare_op = inst.op.rfind("CMP", 0) == 0;
  bool select_op = inst.op == "SEL";

  for (LaneState *lane : active_lanes(warp)) {
    if (inst.op == "MMA" || inst.op == "MMA.I8") {
      throw std::runtime_error("MMA is not implemented in C++ simulator yet");
    }

    if (compare_op) {
      double a = read_operand(device, sm, warp, *lane, inst.operands[1]);
      double b = read_operand(device, sm, warp, *lane, inst.operands[2]);
      write_register(*lane, inst.operands[0], comparison(inst.op, a, b) ? 1.0 : 0.0);
    } else if (select_op) {
      double cond = read_operand(device, sm, warp, *lane, inst.operands[1]);
      double a = read_operand(device, sm, warp, *lane, inst.operands[2]);
      double b = read_operand(device, sm, warp, *lane, inst.operands[3]);
      write_register(*lane, inst.operands[0], cond != 0 ? a : b);
    } else if (memory_load) {
      double addr = read_operand(device, sm, warp, *lane, inst.operands[1]);
      ByteMemory &memory =
          is_shared_memory_op(inst.op) ? sm.shared_memory : device.global_memory;
      double value = read_typed(memory, memory_type(inst.op), static_cast<int64_t>(addr));
      write_register(*lane, inst.operands[0], value);
      if (is_global_memory_op(inst.op))
        device.profile.global_load_bytes += 4;
      else
        device.profile.shared_load_bytes += 4;
    } else if (memory_store) {
      double value = read_operand(device, sm, warp, *lane, inst.operands[1]);
      double addr = read_operand(device, sm, warp, *lane, inst.operands[0]);
      ByteMemory &memory =
          is_shared_memory_op(inst.op) ? sm.shared_memory : device.global_memory;
      write_typed(memory, memory_type(inst.op), static_cast<int64_t>(addr), value);
      if (is_global_memory_op(inst.op))
        device.profile.global_store_bytes += 4;
      else
        device.profile.shared_store_bytes += 4;
    } else {
      if (inst.operands.empty() ||
          inst.operands[0].kind != OperandKind::Reg)
        throw std::runtime_error("instruction has no register destination");
      double a = read_operand(device, sm, warp, *lane, inst.operands[1]);
      double b = inst.operands.size() > 2
                     ? read_operand(device, sm, warp, *lane, inst.operands[2])
                     : 0;
      double c = inst.operands.size() > 3
                     ? read_operand(device, sm, warp, *lane, inst.operands[3])
                     : 0;
      write_register(*lane, inst.operands[0], alu(inst.op, a, b, c));
    }
  }

  device.profile.issued_instructions += 1;
  warp.pc += 1;
}

void populate_sm(SM &sm, int block_linear_id, int block_threads,
                 const std::vector<int> &grid_dim,
                 const std::vector<int> &block_dim, int register_count) {
  sm.warps.clear();
  sm.block_linear_id = block_linear_id;
  sm.block_idx = unflatten_index(block_linear_id, grid_dim);
  int warp_count = (block_threads + 31) / 32;
  for (int warp_index = 0; warp_index < warp_count; ++warp_index) {
    WarpState warp;
    warp.warp_id = warp_index;
    int lane_count = std::min(32, block_threads - warp_index * 32);
    for (int lane_id = 0; lane_id < lane_count; ++lane_id) {
      LaneState lane;
      lane.lane_id = lane_id;
      lane.regs.assign(register_count, 0.0);
      lane.active = true;
      warp.lanes.push_back(std::move(lane));
    }
    warp.active_mask =
        lane_count == 32 ? 0xffffffffu : ((1u << lane_count) - 1u);
    sm.warps.push_back(std::move(warp));
  }
}

} // namespace

Device create_device(size_t global_size, size_t shared_size, int sm_count,
                     int register_count) {
  Device device;
  device.global_memory = ByteMemory(global_size, "global");
  for (int i = 0; i < sm_count; ++i)
    device.sms.push_back(SM{i, ByteMemory(shared_size, "shared" + std::to_string(i))});
  device.register_count = register_count;
  return device;
}

std::vector<int> unflatten_index(int index, const std::vector<int> &shape) {
  std::vector<int> result(shape.size(), 0);
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    result[i] = index % shape[i];
    index /= shape[i];
  }
  return result;
}

ProfileCounters launch(Device &device, const std::string &asm_text,
                       const LaunchConfig &config) {
  Program program = parse_asm(asm_text);
  device.grid_dim = config.grid_dim;
  device.block_dim = config.block_dim;
  device.profile = ProfileCounters{};

  int block_threads = 1;
  for (int dim : device.block_dim)
    block_threads *= dim;
  int total_blocks = 1;
  for (int dim : device.grid_dim)
    total_blocks *= dim;

  for (int wave_start = 0; wave_start < total_blocks;
       wave_start += static_cast<int>(device.sms.size())) {
    int blocks_this_wave =
        std::min<int>(static_cast<int>(device.sms.size()),
                      total_blocks - wave_start);
    for (int sm_index = 0; sm_index < static_cast<int>(device.sms.size());
         ++sm_index) {
      SM &sm = device.sms[sm_index];
      if (sm_index < blocks_this_wave)
        populate_sm(sm, wave_start + sm_index, block_threads, device.grid_dim,
                    device.block_dim, device.register_count);
      else
        sm.warps.clear();
    }

    bool all_exited = false;
    while (!all_exited) {
      all_exited = true;
      ++device.clock;
      ++device.profile.cycles;
      for (SM &sm : device.sms) {
        for (WarpState &warp : sm.warps) {
          if (warp.exited)
            continue;
          all_exited = false;
          if (!warp.waiting_at_barrier)
            execute_instruction(device, sm, warp, program);
        }
      }
    }
  }
  return device.profile;
}

} // namespace cpp_sim
