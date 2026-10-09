#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cpp_sim {

class MemoryError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class ByteMemory {
public:
  ByteMemory(size_t size, std::string name = "memory");

  size_t size() const { return data_.size(); }
  std::vector<uint8_t> &bytes() { return data_; }
  const std::vector<uint8_t> &bytes() const { return data_; }

  void check(int64_t addr, size_t count) const;

  float read_f32(int64_t addr) const;
  void write_f32(int64_t addr, float value);
  float read_f16(int64_t addr) const;
  void write_f16(int64_t addr, float value);
  float read_bf16(int64_t addr) const;
  void write_bf16(int64_t addr, float value);
  float read_f8_e4m3(int64_t addr) const;
  void write_f8_e4m3(int64_t addr, float value);
  float read_f8_e5m2(int64_t addr) const;
  void write_f8_e5m2(int64_t addr, float value);
  int32_t read_i8(int64_t addr) const;
  void write_i8(int64_t addr, int32_t value);
  uint8_t read_u8(int64_t addr) const;
  void write_u8(int64_t addr, uint8_t value);
  int32_t read_i32(int64_t addr) const;
  void write_i32(int64_t addr, int32_t value);
  int32_t read_i4(int64_t addr) const;
  void write_i4(int64_t addr, int32_t value);
  uint8_t read_u4(int64_t addr) const;
  void write_u4(int64_t addr, uint8_t value);

private:
  std::vector<uint8_t> data_;
  std::string name_;
};

} // namespace cpp_sim
