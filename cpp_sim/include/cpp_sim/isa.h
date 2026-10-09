#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cpp_sim {

enum class OperandKind { Reg, Imm, Mem, Label };

struct Operand {
  OperandKind kind = OperandKind::Imm;
  std::string text;
  std::string base;
  bool special = false;
  bool base_special = false;
  int reg_index = -1;
  int base_reg_index = -1;
  int64_t offset = 0;
  double imm = 0.0;
};

struct Instruction {
  std::string op;
  std::vector<Operand> operands;
};

struct Program {
  std::vector<Instruction> instructions;
  std::unordered_map<std::string, int> labels;
};

Program parse_asm(const std::string &text);

bool is_special_register(const std::string &name);
bool is_register_name(const std::string &name);

} // namespace cpp_sim
