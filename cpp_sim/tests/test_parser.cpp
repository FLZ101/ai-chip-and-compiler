#include "cpp_sim/isa.h"

#include <cassert>
#include <iostream>

int main() {
  cpp_sim::Program program = cpp_sim::parse_asm(
      "start:\n"
      "MOV R0, ctaid.x\n"
      "LDG R1, [R0 + 16]\n"
      "BRA start\n");
  assert(program.instructions.size() == 3);
  assert(program.labels.at("start") == 0);
  assert(program.instructions[1].operands[1].kind == cpp_sim::OperandKind::Mem);
  assert(program.instructions[1].operands[1].offset == 16);
  assert(program.instructions[2].operands[0].kind == cpp_sim::OperandKind::Label);
  std::cout << "parser ok\n";
  return 0;
}
