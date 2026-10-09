import unittest

import numpy as np

from compiler.codegen import lower_kernel
from compiler.frontends.manual_frontend import ManualFrontend
from compiler.mlir.mgpu_runner import run_mgpu_pipeline
from compiler.model_ir import DType
from hw.core import launch as python_launch
from hw.device import create_device
from hw.isa import parse_asm

from .helpers import (
    allocate_global_buffers,
    make_device,
    read_tensor,
    run_operator_e2e,
    run_software_model_e2e,
)


def _attention_mlir(batch: int, heads: int, seq: int, dim: int, scale: float) -> str:
    """Build a rank-4 attention function with fixed buffer ordering."""
    return f"""module {{
  func.func @attention(
      %q: memref<{batch}x{heads}x{seq}x{dim}xf32>, %k: memref<{batch}x{heads}x{seq}x{dim}xf32>,
      %v: memref<{batch}x{heads}x{seq}x{dim}xf32>, %mask: memref<{batch}x{heads}x{seq}x{seq}xf32>,
      %out: memref<{batch}x{heads}x{seq}x{dim}xf32>) {{
    %c1 = arith.constant 1 : index
    mgpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) {{
      mgpu.attention %q, %k, %v, %mask, %out {{scale = {scale} : f32, masked}} : memref<{batch}x{heads}x{seq}x{dim}xf32>, memref<{batch}x{heads}x{seq}x{dim}xf32>, memref<{batch}x{heads}x{seq}x{dim}xf32>, memref<{batch}x{heads}x{seq}x{seq}xf32>, memref<{batch}x{heads}x{seq}x{dim}xf32>
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
        _attention_mlir(1, 1, seq, dim, scale),
        pipeline=("builtin.module(func.func(" "mgpu-to-isa{emit-attention-softmax}))"),
    )
    program = parse_asm(result.code)
    python_launch(device, program, grid_dim=result.grid, block_dim=result.block)
    return device


def _run_full_attention(q, k, v, mask):
    """Run the complete tiled attention kernel and return its output."""
    seq = q.shape[2]
    dim = q.shape[3]
    scale = 1.0 / np.sqrt(dim)
    out = np.zeros_like(q)

    device = create_device(global_size=1 << 22, sm_count=4, shared_size=1 << 16)
    layout = allocate_global_buffers(
        device,
        {"q": q, "k": k, "v": v, "mask": mask, "out": out},
    )

    result = run_mgpu_pipeline(
        _attention_mlir(1, 1, seq, dim, scale),
        pipeline=("builtin.module(func.func(" "mgpu-to-isa{emit-attention-tiled}))"),
    )
    program = parse_asm(result.code)
    python_launch(device, program, grid_dim=result.grid, block_dim=result.block)
    return read_tensor(device, layout["out"])


def _run_mapped_full_attention(q, k, v, mask):
    """Run the block-mapped full attention kernel and return its output."""
    batch, heads, seq, dim = q.shape
    scale = 1.0 / np.sqrt(dim)
    out = np.zeros_like(q)

    device = create_device(global_size=1 << 22, sm_count=4, shared_size=1 << 16)
    layout = allocate_global_buffers(
        device,
        {"q": q, "k": k, "v": v, "mask": mask, "out": out},
    )

    result = run_mgpu_pipeline(
        _attention_mlir(batch, heads, seq, dim, scale),
        pipeline=("builtin.module(func.func(" "mgpu-to-isa{emit-attention-tiled}))"),
    )
    program = parse_asm(result.code)
    python_launch(device, program, grid_dim=result.grid, block_dim=result.block)
    return read_tensor(device, layout["out"]), result.grid


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


def _attention_reference(q, k, v, scale, mask):
    weights = _softmax_reference(q, k, scale, mask)
    return np.matmul(weights, v)


class TestTiledAttentionKernels(unittest.TestCase):
    """Numerical coverage for the tiled MMA attention stages."""

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

    def _check_full_attention(self, seq: int, dim: int):
        rng = np.random.default_rng(77)
        q = rng.normal(size=(1, 1, seq, dim)).astype(np.float32)
        k = rng.normal(size=(1, 1, seq, dim)).astype(np.float32)
        v = rng.normal(size=(1, 1, seq, dim)).astype(np.float32)
        mask = np.tril(np.ones((seq, seq), dtype=np.float32)).reshape(1, 1, seq, seq)
        scale = 1.0 / np.sqrt(dim)

        actual = _run_full_attention(q, k, v, mask)
        expected = _attention_reference(
            q.reshape(seq, dim),
            k.reshape(seq, dim),
            v.reshape(seq, dim),
            scale,
            mask.reshape(seq, seq),
        ).reshape(1, 1, seq, dim)
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)

    def test_full_attention_small(self):
        self._check_full_attention(seq=8, dim=64)

    def test_full_attention_tiled(self):
        self._check_full_attention(seq=32, dim=32)

    def test_full_attention_padded_head_dim(self):
        self._check_full_attention(seq=8, dim=20)

    def _check_mapped_attention(self, batch: int, heads: int, seq: int, dim: int):
        rng = np.random.default_rng(123)
        q = rng.normal(size=(batch, heads, seq, dim)).astype(np.float32)
        k = rng.normal(size=(batch, heads, seq, dim)).astype(np.float32)
        v = rng.normal(size=(batch, heads, seq, dim)).astype(np.float32)
        mask = np.tril(np.ones((seq, seq), dtype=np.float32)).reshape(1, 1, seq, seq)
        mask = np.broadcast_to(mask, (batch, heads, seq, seq)).copy()
        scale = 1.0 / np.sqrt(dim)

        actual, grid = _run_mapped_full_attention(q, k, v, mask)
        expected_blocks = batch * heads * ((seq + 15) // 16)
        self.assertEqual(grid, (expected_blocks, 1, 1))

        scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
        scores = np.where(mask == 0, -np.inf, scores)
        shifted = scores - np.max(scores, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        weights = exp / np.sum(exp, axis=-1, keepdims=True)
        expected = np.matmul(weights, v)
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)

    def test_mapped_attention_multi_batch_head(self):
        self._check_mapped_attention(batch=2, heads=3, seq=8, dim=16)

    def test_mapped_attention_multi_query_tile(self):
        self._check_mapped_attention(batch=1, heads=2, seq=20, dim=24)

    def test_attention_mgpu_matches_software(self):
        rng = np.random.default_rng(321)
        q = rng.normal(size=(1, 1, 8, 16)).astype(np.float32)
        k = rng.normal(size=(1, 1, 8, 16)).astype(np.float32)
        v = rng.normal(size=(1, 1, 8, 16)).astype(np.float32)
        mask = np.tril(np.ones((8, 8), dtype=np.float32)).reshape(1, 1, 8, 8)
        scale = 1.0 / np.sqrt(16.0)

        frontend = ManualFrontend("attention_mgpu_vs_software")
        qv = frontend.add_input("q", DType.F32, q.shape)
        kv = frontend.add_input("k", DType.F32, k.shape)
        vv = frontend.add_input("v", DType.F32, v.shape)
        mv = frontend.add_input("mask", DType.F32, mask.shape)
        frontend.output(frontend.masked_attention(qv, kv, vv, mv, scale=scale))
        graph = frontend.build()
        inputs = {"q": q, "k": k, "v": v, "mask": mask}

        software = run_software_model_e2e(
            make_device(global_size=1 << 22, sm_count=4),
            graph,
            inputs,
        )
        mgpu = run_operator_e2e(
            make_device(global_size=1 << 22, sm_count=4),
            graph,
            inputs,
            lowering=lower_kernel,
        )
        np.testing.assert_allclose(
            mgpu.outputs[0],
            software.outputs[0],
            rtol=1e-3,
            atol=1e-3,
        )


if __name__ == "__main__":
    unittest.main()
