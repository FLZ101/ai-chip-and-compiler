#include "cpp_sim/isa.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>

namespace cpp_sim {

namespace {

const std::unordered_map<std::string, bool> &special_registers() {
  static const std::unordered_map<std::string, bool> regs = {
      {"tid.x", true},   {"tid.y", true},   {"tid.z", true},
      {"ntid.x", true},  {"ntid.y", true},  {"ntid.z", true},
      {"ctaid.x", true}, {"ctaid.y", true}, {"ctaid.z", true},
      {"nctaid.x", true},{"nctaid.y", true},{"nctaid.z", true},
      {"laneid", true},  {"warpid", true},  {"clock", true},
  };
  return regs;
}

bool looks_like_label(const std::string &token) {
  if (token.empty() || !std::isalpha(static_cast<unsigned char>(token[0])))
    return false;
  for (char c : token) {
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.'))
      return false;
  }
  return true;
}

std::vector<std::string> split_operands(const std::string &text) {
  std::vector<std::string> parts;
  int depth = 0;
  std::string current;
  for (char c : text) {
    if (c == '[')
      ++depth;
    else if (c == ']')
      --depth;
    if (c == ',' && depth == 0) {
      parts.push_back(current);
      current.clear();
    } else {
      current.push_back(c);
    }
  }
  if (!current.empty())
    parts.push_back(current);
  return parts;
}

Operand parse_operand(std::string token) {
  while (!token.empty() &&
         std::isspace(static_cast<unsigned char>(token.front())))
    token.erase(token.begin());
  while (!token.empty() &&
         std::isspace(static_cast<unsigned char>(token.back())))
    token.pop_back();
  if (token.empty())
    throw std::invalid_argument("empty operand");

  if (token.front() == '[' && token.back() == ']') {
    std::string inner = token.substr(1, token.size() - 2);
    while (!inner.empty() &&
           std::isspace(static_cast<unsigned char>(inner.front())))
      inner.erase(inner.begin());
    while (!inner.empty() &&
           std::isspace(static_cast<unsigned char>(inner.back())))
      inner.pop_back();

    Operand operand;
    operand.kind = OperandKind::Mem;
    if (inner.empty()) {
      operand.base = "RZ";
      operand.offset = 0;
      return operand;
    }

    size_t plus = inner.find('+');
    size_t minus = inner.find('-');
    if (plus != std::string::npos) {
      operand.base = inner.substr(0, plus);
      operand.offset = static_cast<int64_t>(
          std::strtoll(inner.c_str() + plus + 1, nullptr, 0));
    } else if (minus != std::string::npos) {
      operand.base = inner.substr(0, minus);
      operand.offset = -static_cast<int64_t>(
          std::strtoll(inner.c_str() + minus + 1, nullptr, 0));
    } else {
      operand.base = inner;
      operand.offset = 0;
    }
    operand.base.erase(
        std::remove_if(operand.base.begin(), operand.base.end(),
                       [](unsigned char c) { return std::isspace(c); }),
        operand.base.end());
    return operand;
  }

  if (is_register_name(token)) {
    Operand operand;
    operand.kind = OperandKind::Reg;
    operand.text = token;
    return operand;
  }

  try {
    Operand operand;
    operand.kind = OperandKind::Imm;
    operand.imm = static_cast<double>(std::stoll(token, nullptr, 0));
    return operand;
  } catch (...) {
  }

  try {
    Operand operand;
    operand.kind = OperandKind::Imm;
    operand.imm = std::stod(token);
    return operand;
  } catch (...) {
  }

  if (looks_like_label(token)) {
    Operand operand;
    operand.kind = OperandKind::Label;
    operand.text = token;
    return operand;
  }

  throw std::invalid_argument("invalid operand: " + token);
}

} // namespace

bool is_special_register(const std::string &name) {
  return special_registers().count(name) != 0;
}

bool is_register_name(const std::string &name) {
  if (is_special_register(name))
    return true;
  if (name.size() < 2 || name[0] != 'R')
    return false;
  for (size_t i = 1; i < name.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(name[i])))
      return false;
  }
  return true;
}

Program parse_asm(const std::string &text) {
  Program program;
  std::vector<std::string> lines;
  std::string current;
  for (char c : text) {
    if (c == '\n') {
      lines.push_back(current);
      current.clear();
    } else {
      current.push_back(c);
    }
  }
  if (!current.empty())
    lines.push_back(current);

  for (const std::string &raw_line : lines) {
    std::string line = raw_line;
    size_t comment = line.find("//");
    if (comment != std::string::npos)
      line.erase(comment);
    while (!line.empty() &&
           std::isspace(static_cast<unsigned char>(line.front())))
      line.erase(line.begin());
    while (!line.empty() &&
           std::isspace(static_cast<unsigned char>(line.back())))
      line.pop_back();
    if (line.empty())
      continue;

    if (line.back() == ':') {
      std::string label = line.substr(0, line.size() - 1);
      program.labels[label] = static_cast<int>(program.instructions.size());
      continue;
    }

    size_t space = line.find_first_of(" \t");
    if (space == std::string::npos) {
      Instruction inst;
      inst.op = line;
      program.instructions.push_back(std::move(inst));
      continue;
    }

    Instruction inst;
    inst.op = line.substr(0, space);
    std::string operand_text = line.substr(space);
    while (!operand_text.empty() &&
           std::isspace(static_cast<unsigned char>(operand_text.front())))
      operand_text.erase(operand_text.begin());
    for (const std::string &part : split_operands(operand_text))
      inst.operands.push_back(parse_operand(part));
    program.instructions.push_back(std::move(inst));
  }

  for (const auto &[label, pc] : program.labels) {
    (void)pc;
  }
  return program;
}

} // namespace cpp_sim
