#include "cpp_sim/software_kernels.h"

#include <cmath>
#include <stdexcept>

namespace cpp_sim {
namespace {

void run_conv2d(ByteMemory &memory, const SoftwareKernelRequest &r) {
  if (r.input_shape.size() != 4 || r.weight_shape.size() != 4 ||
      r.output_shape.size() != 4)
    throw std::runtime_error("conv2d requires NCHW tensors");
  const int64_t n = r.input_shape[0];
  const int64_t c = r.input_shape[1];
  const int64_t h = r.input_shape[2];
  const int64_t w = r.input_shape[3];
  const int64_t f = r.weight_shape[0];
  const int64_t kh = r.weight_shape[2];
  const int64_t kw = r.weight_shape[3];
  const int64_t on = r.output_shape[0];
  const int64_t of = r.output_shape[1];
  const int64_t oh = r.output_shape[2];
  const int64_t ow = r.output_shape[3];
  const int64_t sy = r.strides[0];
  const int64_t sx = r.strides[1];
  const int64_t py = r.paddings[0];
  const int64_t px = r.paddings[1];
  const int64_t dy = r.dilations[0];
  const int64_t dx = r.dilations[1];
  if (n != 1 || on != 1 || f != of)
    throw std::runtime_error("conv2d software kernel supports batch=1 only");

  for (int64_t ff = 0; ff < f; ++ff) {
    for (int64_t oy = 0; oy < oh; ++oy) {
      for (int64_t ox = 0; ox < ow; ++ox) {
        float acc = 0.0f;
        for (int64_t cc = 0; cc < c; ++cc) {
          for (int64_t ky = 0; ky < kh; ++ky) {
            const int64_t iy = oy * sy + ky * dy - py;
            if (iy < 0 || iy >= h)
              continue;
            for (int64_t kx = 0; kx < kw; ++kx) {
              const int64_t ix = ox * sx + kx * dx - px;
              if (ix < 0 || ix >= w)
                continue;
              const int64_t input_addr =
                  r.input_offset + ((cc * h + iy) * w + ix) * 4;
              const int64_t weight_addr =
                  r.weight_offset + ((ff * c + cc) * kh + ky) * kw * 4 +
                  kx * 4;
              acc += memory.read_f32(input_addr) * memory.read_f32(weight_addr);
            }
          }
        }
        if (r.has_bias)
          acc += memory.read_f32(r.bias_offset + ff * 4);
        const int64_t output_addr =
            r.output_offset + ((ff * oh + oy) * ow + ox) * 4;
        memory.write_f32(output_addr, acc);
      }
    }
  }
}

void run_matmul(ByteMemory &memory, const SoftwareKernelRequest &r) {
  if (r.input_shape.size() != 2 || r.weight_shape.size() != 2 ||
      r.output_shape.size() != 2)
    throw std::runtime_error("matmul requires rank-2 tensors");
  const int64_t m = r.input_shape[0];
  const int64_t k = r.input_shape[1];
  const int64_t n = r.weight_shape[1];
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int64_t inner = 0; inner < k; ++inner) {
        acc += memory.read_f32(r.input_offset + (row * k + inner) * 4) *
               memory.read_f32(r.weight_offset + (inner * n + col) * 4);
      }
      if (r.has_bias)
        acc += memory.read_f32(r.bias_offset + col * 4);
      memory.write_f32(r.output_offset + (row * n + col) * 4, acc);
    }
  }
}

void run_linear(ByteMemory &memory, const SoftwareKernelRequest &r) {
  if (r.input_shape.size() != 2 || r.weight_shape.size() != 2 ||
      r.output_shape.size() != 2)
    throw std::runtime_error("linear requires rank-2 tensors");
  const int64_t m = r.input_shape[0];
  const int64_t k = r.input_shape[1];
  const int64_t n = r.weight_shape[0];
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t out = 0; out < n; ++out) {
      float acc = 0.0f;
      for (int64_t inner = 0; inner < k; ++inner) {
        acc += memory.read_f32(r.input_offset + (row * k + inner) * 4) *
               memory.read_f32(r.weight_offset + (out * k + inner) * 4);
      }
      if (r.has_bias)
        acc += memory.read_f32(r.bias_offset + out * 4);
      memory.write_f32(r.output_offset + (row * n + out) * 4, acc);
    }
  }
}

void run_relu(ByteMemory &memory, const SoftwareKernelRequest &r) {
  int64_t numel = 1;
  for (int64_t dim : r.input_shape)
    numel *= dim;
  for (int64_t i = 0; i < numel; ++i)
    memory.write_f32(r.output_offset + i * 4,
                     std::max(0.0f, memory.read_f32(r.input_offset + i * 4)));
}

void run_add(ByteMemory &memory, const SoftwareKernelRequest &r) {
  int64_t numel = 1;
  for (int64_t dim : r.input_shape)
    numel *= dim;
  for (int64_t i = 0; i < numel; ++i) {
    memory.write_f32(
        r.output_offset + i * 4,
        memory.read_f32(r.input_offset + i * 4) +
            memory.read_f32(r.input2_offset + i * 4));
  }
}

void run_batch_norm(ByteMemory &memory, const SoftwareKernelRequest &r) {
  if (r.input_shape.size() != 4 || r.weight_shape.size() != 1 ||
      r.output_shape.size() != 4)
    throw std::runtime_error("batch_norm requires NCHW input and C weights");
  const int64_t c = r.input_shape[1];
  const int64_t h = r.input_shape[2];
  const int64_t w = r.input_shape[3];
  const int64_t hw = h * w;
  for (int64_t cc = 0; cc < c; ++cc) {
    float mean = memory.read_f32(r.mean_offset + cc * 4);
    float var = memory.read_f32(r.var_offset + cc * 4);
    float rstd = 1.0f / std::sqrt(var + static_cast<float>(r.epsilon));
    float scale = memory.read_f32(r.weight_offset + cc * 4);
    float bias = memory.read_f32(r.bias_offset + cc * 4);
    for (int64_t i = 0; i < hw; ++i) {
      int64_t in_addr = r.input_offset + (cc * hw + i) * 4;
      float x = memory.read_f32(in_addr);
      float y = (x - mean) * rstd * scale + bias;
      memory.write_f32(r.output_offset + (cc * hw + i) * 4, y);
    }
  }
}

void run_global_avg_pool(ByteMemory &memory, const SoftwareKernelRequest &r) {
  if (r.input_shape.size() != 4 || r.output_shape.size() != 4)
    throw std::runtime_error("global_avg_pool requires rank-4 NCHW");
  const int64_t c = r.input_shape[1];
  const int64_t h = r.input_shape[2];
  const int64_t w = r.input_shape[3];
  const int64_t hw = h * w;
  for (int64_t cc = 0; cc < c; ++cc) {
    float sum = 0.0f;
    for (int64_t i = 0; i < hw; ++i)
      sum += memory.read_f32(r.input_offset + (cc * hw + i) * 4);
    memory.write_f32(r.output_offset + cc * 4, sum / static_cast<float>(hw));
  }
}

void run_pool2d(ByteMemory &memory, const SoftwareKernelRequest &r, bool max_pool) {
  if (r.input_shape.size() != 4 || r.output_shape.size() != 4)
    throw std::runtime_error("pooling requires rank-4 NCHW");
  const int64_t c = r.input_shape[1];
  const int64_t h = r.input_shape[2];
  const int64_t w = r.input_shape[3];
  const int64_t oh = r.output_shape[2];
  const int64_t ow = r.output_shape[3];
  const int64_t kh = r.kernel[0];
  const int64_t kw = r.kernel[1];
  const int64_t sy = r.strides[0];
  const int64_t sx = r.strides[1];
  const int64_t py = r.paddings[0];
  const int64_t px = r.paddings[1];
  const int64_t dy = r.dilations[0];
  const int64_t dx = r.dilations[1];
  for (int64_t cc = 0; cc < c; ++cc) {
    for (int64_t oy = 0; oy < oh; ++oy) {
      for (int64_t ox = 0; ox < ow; ++ox) {
        float value = max_pool ? -1.0e30f : 0.0f;
        for (int64_t ky = 0; ky < kh; ++ky) {
          int64_t iy = oy * sy + ky * dy - py;
          if (iy < 0 || iy >= h)
            continue;
          for (int64_t kx = 0; kx < kw; ++kx) {
            int64_t ix = ox * sx + kx * dx - px;
            if (ix < 0 || ix >= w)
              continue;
            float x = memory.read_f32(
                r.input_offset + ((cc * h + iy) * w + ix) * 4);
            value = max_pool ? std::max(value, x) : value + x;
          }
        }
        if (!max_pool)
          value /= static_cast<float>(kh * kw);
        memory.write_f32(
            r.output_offset + ((cc * oh + oy) * ow + ox) * 4, value);
      }
    }
  }
}

void run_pad(ByteMemory &memory, const SoftwareKernelRequest &r) {
  if (r.input_shape.size() != 4 || r.output_shape.size() != 4)
    throw std::runtime_error("pad requires rank-4 NCHW");
  const int64_t c = r.input_shape[1];
  const int64_t h = r.input_shape[2];
  const int64_t w = r.input_shape[3];
  const int64_t oh = r.output_shape[2];
  const int64_t ow = r.output_shape[3];
  const int64_t py = r.paddings[0];
  const int64_t px = r.paddings[1];
  const float value = static_cast<float>(r.pad_value);
  for (int64_t cc = 0; cc < c; ++cc) {
    for (int64_t oy = 0; oy < oh; ++oy) {
      for (int64_t ox = 0; ox < ow; ++ox) {
        int64_t iy = oy - py;
        int64_t ix = ox - px;
        float x = value;
        if (iy >= 0 && iy < h && ix >= 0 && ix < w)
          x = memory.read_f32(
              r.input_offset + ((cc * h + iy) * w + ix) * 4);
        memory.write_f32(
            r.output_offset + ((cc * oh + oy) * ow + ox) * 4, x);
      }
    }
  }
}

void run_reshape(ByteMemory &memory, const SoftwareKernelRequest &r) {
  int64_t numel = 1;
  for (int64_t dim : r.input_shape)
    numel *= dim;
  for (int64_t i = 0; i < numel; ++i)
    memory.write_f32(r.output_offset + i * 4,
                     memory.read_f32(r.input_offset + i * 4));
}

} // namespace

bool run_software_kernel(ByteMemory &memory, const SoftwareKernelRequest &request) {
  if (request.op == "conv2d") {
    run_conv2d(memory, request);
    return true;
  }
  if (request.op == "matmul") {
    run_matmul(memory, request);
    return true;
  }
  if (request.op == "linear") {
    run_linear(memory, request);
    return true;
  }
  if (request.op == "relu") {
    run_relu(memory, request);
    return true;
  }
  if (request.op == "add") {
    run_add(memory, request);
    return true;
  }
  if (request.op == "batch_norm") {
    run_batch_norm(memory, request);
    return true;
  }
  if (request.op == "global_avg_pool") {
    run_global_avg_pool(memory, request);
    return true;
  }
  if (request.op == "max_pool2d") {
    run_pool2d(memory, request, true);
    return true;
  }
  if (request.op == "avg_pool2d") {
    run_pool2d(memory, request, false);
    return true;
  }
  if (request.op == "pad") {
    run_pad(memory, request);
    return true;
  }
  if (request.op == "reshape") {
    run_reshape(memory, request);
    return true;
  }
  return false;
}

} // namespace cpp_sim
