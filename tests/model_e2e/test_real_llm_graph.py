"""Model IR graph construction smoke test for a GPT-2-like decoder."""

from __future__ import annotations

import unittest

from compiler.frontends.manual_frontend import ManualFrontend
from compiler.model_ir import DType


def build_minimal_gpt2_graph():
    vocab = 20
    hidden = 8
    seq = 3
    frontend = ManualFrontend("minimal_gpt2")

    input_ids = frontend.add_input("input_ids", DType.I32, (1, seq))
    token_embed = frontend.add_input("token_embed", DType.F32, (vocab, hidden))
    pos_embed = frontend.add_input("pos_embed", DType.F32, (1, seq, hidden))
    q_weight = frontend.add_input("q_weight", DType.F32, (hidden, hidden))
    q_bias = frontend.add_input("q_bias", DType.F32, (hidden,))
    k_weight = frontend.add_input("k_weight", DType.F32, (hidden, hidden))
    k_bias = frontend.add_input("k_bias", DType.F32, (hidden,))
    v_weight = frontend.add_input("v_weight", DType.F32, (hidden, hidden))
    v_bias = frontend.add_input("v_bias", DType.F32, (hidden,))
    out_weight = frontend.add_input("out_weight", DType.F32, (hidden, hidden))
    out_bias = frontend.add_input("out_bias", DType.F32, (hidden,))
    ln_weight = frontend.add_input("ln_weight", DType.F32, (hidden,))
    ln_bias = frontend.add_input("ln_bias", DType.F32, (hidden,))
    lm_head = frontend.add_input("lm_head", DType.F32, (vocab, hidden))

    tokens = frontend.embedding_gather(input_ids, token_embed)
    hidden_states = frontend.add(tokens, pos_embed)
    hidden_states = frontend.layer_norm(hidden_states, ln_weight, ln_bias)

    q = frontend.linear(hidden_states, q_weight, q_bias)
    k = frontend.linear(hidden_states, k_weight, k_bias)
    v = frontend.linear(hidden_states, v_weight, v_bias)
    q = frontend.reshape(q, (1, 2, seq, 4))
    k = frontend.reshape(k, (1, 2, seq, 4))
    v = frontend.reshape(v, (1, 2, seq, 4))
    attn = frontend.scaled_dot_product_attention(q, k, v)
    attn = frontend.reshape(attn, (1, seq, hidden))
    attn = frontend.linear(attn, out_weight, out_bias)
    hidden_states = frontend.add(hidden_states, attn)
    hidden_states = frontend.layer_norm(hidden_states, ln_weight, ln_bias)
    logits = frontend.linear(hidden_states, lm_head)
    next_token = frontend.argmax(logits, axis=-1)
    frontend.output(next_token)
    return frontend.build()


class TestRealLLMGraph(unittest.TestCase):
    def test_minimal_gpt2_graph_builds_and_validates(self):
        graph = build_minimal_gpt2_graph()
        graph.validate()
        self.assertEqual(graph.outputs[0].shape, (1, 3))
        self.assertIn("embedding_gather", {node.op.value for node in graph.nodes})
        self.assertIn(
            "scaled_dot_product_attention", {node.op.value for node in graph.nodes}
        )
        self.assertIn("argmax", {node.op.value for node in graph.nodes})


if __name__ == "__main__":
    unittest.main()
