#include "cpp_sim/memory.h"

#include <cassert>
#include <iostream>

int main() {
  cpp_sim::ByteMemory memory(64, "test");
  memory.write_f32(0, 1.5f);
  assert(memory.read_f32(0) == 1.5f);
  memory.write_i4(0, -3);
  memory.write_i4(1, 5);
  assert(memory.read_i4(0) == -3);
  assert(memory.read_i4(1) == 5);
  std::cout << "memory ok\n";
  return 0;
}
