import unittest

import numpy as np

from compiler.codegen import lower_kernel

from examples.transformer import (
    autoregressive_loop,
    decoder_block,
    encoder_block,
    kv_cache_decode,
)
from tests.operator_e2e.helpers import make_device, run_operator_e2e


class TestTransformerModelExamples(unittest.TestCase):
    """Step 017 Transformer/LLM model examples."""

    def test_encoder_block(self):
        x = np.arange(8, dtype=np.float32).reshape(2, 4)
        norm_w = np.array([1.0, 1.5, 0.5, 2.0], dtype=np.float32)
        norm_b = np.array([0.0, 0.1, -0.1, 0.2], dtype=np.float32)

        graph = encoder_block.build_graph()
        result = run_operator_e2e(
            make_device(),
            graph,
            {"x": x, "norm_w": norm_w, "norm_b": norm_b},
            [encoder_block.reference(x, norm_w, norm_b)],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertIn('"layer_norm"', graph.format())

    def test_decoder_block_with_causal_mask(self):
        q = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
        k = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        v = np.array([[[[2.0, 1.0], [0.0, 3.0]]]], dtype=np.float32)

        graph = decoder_block.build_graph()
        result = run_operator_e2e(
            make_device(),
            graph,
            {"q": q, "k": k, "v": v},
            [decoder_block.reference(q, k, v)],
            lowering=lower_kernel,
            atol=1e-5,
        )
        self.assertIn('"causal_mask"', graph.format())
        self.assertIn('"masked_attention"', graph.format())

    def test_kv_cache_decode(self):
        cache = np.arange(8, dtype=np.float32).reshape(1, 1, 4, 2)
        positions = np.array([[2]], dtype=np.float32)

        graph = kv_cache_decode.build_graph()
        result = run_operator_e2e(
            make_device(),
            graph,
            {"cache": cache, "positions": positions},
            [kv_cache_decode.reference(cache, positions)],
            lowering=lower_kernel,
        )
        self.assertIn('"kv_cache_gather"', graph.format())

    def test_autoregressive_loop(self):
        cache = np.arange(8, dtype=np.float32).reshape(1, 1, 4, 2)
        pos0 = np.array([[1]], dtype=np.float32)
        pos1 = np.array([[3]], dtype=np.float32)

        graph = autoregressive_loop.build_graph()
        expected = autoregressive_loop.reference(cache, pos0, pos1)
        result = run_operator_e2e(
            make_device(),
            graph,
            {"cache": cache, "pos0": pos0, "pos1": pos1},
            expected,
            lowering=lower_kernel,
        )
        self.assertEqual(len(result.outputs), 2)


if __name__ == "__main__":
    unittest.main()
