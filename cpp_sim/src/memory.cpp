#include "cpp_sim/memory.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace cpp_sim {

namespace {

float bits_to_float(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint32_t float_to_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

uint8_t f8_e4m3_encode(float value) {
  if (std::isnan(value))
    return 0x7f;
  if (value >= 448.0f)
    return 0x7e;
  if (value <= -448.0f)
    return 0xfe;
  if (std::abs(value) < 1.0f / 64.0f)
    return 0;
  uint32_t bits = float_to_bits(value);
  uint32_t sign = (bits >> 16) & 0x8000u;
  int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127;
  uint32_t mantissa = bits & 0x7fffffu;
  mantissa |= 0x800000u;
  int e4 = exponent + 7;
  if (e4 < 0)
    e4 = 0;
  if (e4 > 15)
    e4 = 15;
  uint32_t m3 = (mantissa >> 20) & 0x7u;
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(e4) << 3) | m3);
}

float f8_e4m3_decode(uint8_t raw) {
  uint32_t sign = (raw & 0x80u) ? 0x80000000u : 0u;
  uint32_t e4 = (raw >> 3) & 0xfu;
  uint32_t m3 = raw & 0x7u;
  int exponent = static_cast<int>(e4) - 7;
  uint32_t mantissa = m3 << 20;
  if (e4 == 0 && m3 == 0)
    return bits_to_float(sign);
  uint32_t biased = static_cast<uint32_t>(exponent + 127) << 23;
  return bits_to_float(sign | biased | mantissa);
}

uint8_t f8_e5m2_encode(float value) {
  if (std::isnan(value))
    return 0x7f;
  if (value >= 57344.0f)
    return 0x7b;
  if (value <= -57344.0f)
    return 0xfb;
  if (std::abs(value) < 1.0f / 16384.0f)
    return 0;
  uint32_t bits = float_to_bits(value);
  uint32_t sign = (bits >> 16) & 0x8000u;
  int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127;
  uint32_t mantissa = bits & 0x7fffffu;
  mantissa |= 0x800000u;
  int e5 = exponent + 15;
  if (e5 < 0)
    e5 = 0;
  if (e5 > 31)
    e5 = 31;
  uint32_t m2 = (mantissa >> 21) & 0x3u;
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(e5) << 2) | m2);
}

float f8_e5m2_decode(uint8_t raw) {
  uint32_t sign = (raw & 0x80u) ? 0x80000000u : 0u;
  uint32_t e5 = (raw >> 2) & 0x1fu;
  uint32_t m2 = raw & 0x3u;
  int exponent = static_cast<int>(e5) - 15;
  uint32_t mantissa = m2 << 21;
  if (e5 == 0 && m2 == 0)
    return bits_to_float(sign);
  uint32_t biased = static_cast<uint32_t>(exponent + 127) << 23;
  return bits_to_float(sign | biased | mantissa);
}

} // namespace

ByteMemory::ByteMemory(size_t size, std::string name)
    : data_(size), name_(std::move(name)) {}

void ByteMemory::check(int64_t addr, size_t count) const {
  if (addr < 0 || count > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      static_cast<uint64_t>(addr) + count > data_.size())
    throw MemoryError(name_ + " access out of bounds");
}

float ByteMemory::read_f32(int64_t addr) const {
  check(addr, 4);
  uint32_t bits = 0;
  std::memcpy(&bits, data_.data() + addr, sizeof(bits));
  return bits_to_float(bits);
}

void ByteMemory::write_f32(int64_t addr, float value) {
  check(addr, 4);
  uint32_t bits = float_to_bits(value);
  std::memcpy(data_.data() + addr, &bits, sizeof(bits));
}

float ByteMemory::read_f16(int64_t addr) const {
  check(addr, 2);
  uint16_t raw = 0;
  std::memcpy(&raw, data_.data() + addr, sizeof(raw));
  uint32_t sign = (raw & 0x8000u) ? 0x80000000u : 0u;
  uint32_t exp = (raw >> 10) & 0x1fu;
  uint32_t man = raw & 0x3ffu;
  uint32_t bits = 0;
  if (exp == 0) {
    if (man == 0)
      return bits_to_float(sign);
    exp = 1;
    while ((man & 0x400u) == 0) {
      man <<= 1;
      --exp;
    }
    man &= 0x3ffu;
    bits = sign | ((exp + 112) << 23) | (man << 13);
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (man << 13);
  } else {
    bits = sign | ((exp + 112) << 23) | (man << 13);
  }
  return bits_to_float(bits);
}

void ByteMemory::write_f16(int64_t addr, float value) {
  check(addr, 2);
  uint32_t bits = float_to_bits(value);
  uint32_t sign = bits >> 16 & 0x8000u;
  int exp = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t man = bits & 0x7fffffu;
  if (exp <= 0) {
    if (exp < -10)
      exp = -11;
    man = (man | 0x800000u) >> (1 - exp);
    exp = 0;
  } else if (exp >= 31) {
    exp = 31;
    man = 0;
  }
  uint16_t raw = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                                       ((man >> 13) & 0x3ffu));
  std::memcpy(data_.data() + addr, &raw, sizeof(raw));
}

float ByteMemory::read_bf16(int64_t addr) const {
  check(addr, 2);
  uint16_t raw = 0;
  std::memcpy(&raw, data_.data() + addr, sizeof(raw));
  uint32_t bits = static_cast<uint32_t>(raw) << 16;
  return bits_to_float(bits);
}

void ByteMemory::write_bf16(int64_t addr, float value) {
  check(addr, 2);
  uint32_t bits = float_to_bits(value);
  uint16_t raw = static_cast<uint16_t>((bits + 0x8000u) >> 16);
  std::memcpy(data_.data() + addr, &raw, sizeof(raw));
}

float ByteMemory::read_f8_e4m3(int64_t addr) const {
  check(addr, 1);
  return f8_e4m3_decode(data_[addr]);
}

void ByteMemory::write_f8_e4m3(int64_t addr, float value) {
  check(addr, 1);
  data_[addr] = f8_e4m3_encode(value);
}

float ByteMemory::read_f8_e5m2(int64_t addr) const {
  check(addr, 1);
  return f8_e5m2_decode(data_[addr]);
}

void ByteMemory::write_f8_e5m2(int64_t addr, float value) {
  check(addr, 1);
  data_[addr] = f8_e5m2_encode(value);
}

int32_t ByteMemory::read_i8(int64_t addr) const {
  check(addr, 1);
  return static_cast<int8_t>(data_[addr]);
}

void ByteMemory::write_i8(int64_t addr, int32_t value) {
  check(addr, 1);
  data_[addr] = static_cast<uint8_t>(static_cast<int8_t>(value));
}

uint8_t ByteMemory::read_u8(int64_t addr) const {
  check(addr, 1);
  return data_[addr];
}

void ByteMemory::write_u8(int64_t addr, uint8_t value) {
  check(addr, 1);
  data_[addr] = value;
}

int32_t ByteMemory::read_i32(int64_t addr) const {
  check(addr, 4);
  int32_t value = 0;
  std::memcpy(&value, data_.data() + addr, sizeof(value));
  return value;
}

void ByteMemory::write_i32(int64_t addr, int32_t value) {
  check(addr, 4);
  std::memcpy(data_.data() + addr, &value, sizeof(value));
}

int32_t ByteMemory::read_i4(int64_t addr) const {
  check(addr / 2, 1);
  uint8_t byte = data_[addr / 2];
  uint8_t raw = (byte >> (4 * (addr % 2))) & 0xf;
  return raw & 0x8 ? raw - 16 : raw;
}

void ByteMemory::write_i4(int64_t addr, int32_t value) {
  check(addr / 2, 1);
  if (value < -8 || value > 7)
    throw MemoryError("i4 value out of range");
  uint8_t byte_addr = static_cast<uint8_t>(addr / 2);
  uint8_t shift = static_cast<uint8_t>(4 * (addr % 2));
  uint8_t mask = static_cast<uint8_t>(0xf << shift);
  uint8_t packed = static_cast<uint8_t>((value & 0xf) << shift);
  data_[byte_addr] = (data_[byte_addr] & ~mask) | packed;
}

uint8_t ByteMemory::read_u4(int64_t addr) const {
  check(addr / 2, 1);
  return (data_[addr / 2] >> (4 * (addr % 2))) & 0xf;
}

void ByteMemory::write_u4(int64_t addr, uint8_t value) {
  check(addr / 2, 1);
  if (value > 15)
    throw MemoryError("u4 value out of range");
  uint8_t byte_addr = static_cast<uint8_t>(addr / 2);
  uint8_t shift = static_cast<uint8_t>(4 * (addr % 2));
  uint8_t mask = static_cast<uint8_t>(0xf << shift);
  data_[byte_addr] = (data_[byte_addr] & ~mask) |
                     static_cast<uint8_t>(value << shift);
}

} // namespace cpp_sim
