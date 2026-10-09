#include "cpp_sim/software_kernels.h"

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

} // namespace

bool run_software_kernel(ByteMemory &memory, const SoftwareKernelRequest &request) {
  if (request.op == "conv2d") {
    run_conv2d(memory, request);
    return true;
  }
  return false;
}

} // namespace cpp_sim
