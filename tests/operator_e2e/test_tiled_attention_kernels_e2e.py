import unittest

import numpy as np

from compiler.mlir.mgpu_runner import run_mgpu_pipeline
from hw.core import launch as python_launch
from hw.device import create_device
from hw.isa import parse_asm

from .helpers import allocate_global_buffers


def _attention_mlir(seq: int, dim: int, scale: float) -> str:
    """Build a single rank-4 attention function with fixed buffer ordering."""
    return f"""module {{
  func.func @attention(
      %q: memref<1x1x{seq}x{dim}xf32>, %k: memref<1x1x{seq}x{dim}xf32>,
      %v: memref<1x1x{seq}x{dim}xf32>, %mask: memref<1x1x{seq}x{seq}xf32>,
      %out: memref<1x1x{seq}x{dim}xf32>) {{
    %c1 = arith.constant 1 : index
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) {{
      mgpu.attention %q, %k, %v, %mask, %out {{scale = {scale} : f32, masked}} : memref<1x1x{seq}x{dim}xf32>, memref<1x1x{seq}x{dim}xf32>, memref<1x1x{seq}x{dim}xf32>, memref<1x1x{seq}x{seq}xf32>, memref<1x1x{seq}x{dim}xf32>
      mgpu.return
    }}
    return
  }}
}}"""


def _run_softmax_stage(q, k, v, mask):
    """Run the tiled score + softmax stage and return the Python device."""
    seq = q.shape[2]
    dim = q.shape[3]
    scale = 1.0 / np.sqrt(dim)
    out = np.zeros_like(q)

    device = create_device(global_size=1 << 22, sm_count=4, shared_size=1 << 16)
    allocate_global_buffers(
        device,
        {"q": q, "k": k, "v": v, "mask": mask, "out": out},
    )

    result = run_mgpu_pipeline(
        _attention_mlir(seq, dim, scale),
        pipeline=("builtin.module(func.func(" "mgpu-to-isa{emit-attention-softmax}))"),
    )
    program = parse_asm(result.code)
    python_launch(device, program, grid_dim=result.grid, block_dim=result.block)
    return device


def _read_softmax_weights(device, seq: int) -> np.ndarray:
    """Reconstruct seq x seq softmax weights from padded 16x16 score tiles."""
    tile_count = (seq + 15) // 16
    weights = np.zeros((seq, seq), dtype=np.float32)

    for q_tile in range(tile_count):
        q_start = q_tile * 16
        q_end = min(q_start + 16, seq)
        for k_tile in range(tile_count):
            k_start = k_tile * 16
            k_end = min(k_start + 16, seq)
            tile_offset = (q_tile * tile_count + k_tile) * 16 * 16 * 4
            raw = device.sms[0].shared_memory.read_floats(tile_offset, 16 * 16)
            tile = np.asarray(raw, dtype=np.float32).reshape(16, 16)
            weights[q_start:q_end, k_start:k_end] = tile[
                : q_end - q_start,
                : k_end - k_start,
            ]
    return weights


def _softmax_reference(q, k, scale, mask):
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    if mask is not None:
        scores = np.where(mask == 0, -np.inf, scores)
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class TestTiledAttentionKernels(unittest.TestCase):
    """Numerical coverage for the tiled MMA attention softmax stage."""

    def _check_softmax(self, seq: int, dim: int):
        rng = np.random.default_rng(31)
        q = rng.normal(size=(1, 1, seq, dim)).astype(np.float32)
        k = rng.normal(size=(1, 1, seq, dim)).astype(np.float32)
        v = np.zeros((1, 1, seq, dim), dtype=np.float32)
        mask = np.tril(np.ones((seq, seq), dtype=np.float32)).reshape(1, 1, seq, seq)
        scale = 1.0 / np.sqrt(dim)

        device = _run_softmax_stage(q, k, v, mask)
        actual = _read_softmax_weights(device, seq)
        expected = _softmax_reference(
            q.reshape(seq, dim), k.reshape(seq, dim), scale, mask.reshape(seq, seq)
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)

    def test_masked_softmax_small(self):
        self._check_softmax(seq=8, dim=64)

    def test_masked_softmax_tiled(self):
        self._check_softmax(seq=32, dim=32)


if __name__ == "__main__":
    unittest.main()
