"""Model IR graph construction smoke test for a GPT-2-like decoder."""

from __future__ import annotations

import unittest

from compiler.frontends.manual_frontend import ManualFrontend
from compiler.mlir import MLIRBuilder
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


def build_full_gpt2_graph(
    seq: int = 8,
    vocab: int = 50257,
    hidden: int = 768,
    heads: int = 12,
    layers: int = 12,
    output_logits: bool = False,
):
    head_dim = hidden // heads
    mlp_hidden = 4 * hidden
    frontend = ManualFrontend("full_gpt2")

    input_ids = frontend.add_input("input_ids", DType.I32, (1, seq))
    token_embed = frontend.add_input("token_embed", DType.F32, (vocab, hidden))
    pos_embed = frontend.add_input("pos_embed", DType.F32, (1, seq, hidden))
    causal_mask = frontend.add_input("causal_mask", DType.F32, (1, 1, seq, seq))
    final_ln_weight = frontend.add_input("final_ln_weight", DType.F32, (hidden,))
    final_ln_bias = frontend.add_input("final_ln_bias", DType.F32, (hidden,))
    lm_head = frontend.add_input("lm_head", DType.F32, (vocab, hidden))

    hidden_states = frontend.embedding_gather(input_ids, token_embed)
    hidden_states = frontend.add(hidden_states, pos_embed)

    for layer in range(layers):
        residual = hidden_states
        ln1_w = frontend.add_input(f"h{layer}_ln_1_weight", DType.F32, (hidden,))
        ln1_b = frontend.add_input(f"h{layer}_ln_1_bias", DType.F32, (hidden,))
        q_w = frontend.add_input(f"h{layer}_q_weight", DType.F32, (hidden, hidden))
        q_b = frontend.add_input(f"h{layer}_q_bias", DType.F32, (hidden,))
        k_w = frontend.add_input(f"h{layer}_k_weight", DType.F32, (hidden, hidden))
        k_b = frontend.add_input(f"h{layer}_k_bias", DType.F32, (hidden,))
        v_w = frontend.add_input(f"h{layer}_v_weight", DType.F32, (hidden, hidden))
        v_b = frontend.add_input(f"h{layer}_v_bias", DType.F32, (hidden,))
        out_w = frontend.add_input(f"h{layer}_out_weight", DType.F32, (hidden, hidden))
        out_b = frontend.add_input(f"h{layer}_out_bias", DType.F32, (hidden,))

        hidden_states = frontend.layer_norm(hidden_states, ln1_w, ln1_b)
        q = frontend.linear(hidden_states, q_w, q_b)
        k = frontend.linear(hidden_states, k_w, k_b)
        v = frontend.linear(hidden_states, v_w, v_b)
        q = frontend.reshape(q, (1, seq, heads, head_dim))
        q = frontend.transpose(q, (0, 2, 1, 3))
        k = frontend.reshape(k, (1, seq, heads, head_dim))
        k = frontend.transpose(k, (0, 2, 1, 3))
        v = frontend.reshape(v, (1, seq, heads, head_dim))
        v = frontend.transpose(v, (0, 2, 1, 3))
        attention = frontend.masked_attention(
            q,
            k,
            v,
            causal_mask,
            scale=1.0 / (head_dim**0.5),
        )
        attention = frontend.transpose(attention, (0, 2, 1, 3))
        attention = frontend.reshape(attention, (1, seq, hidden))
        attention = frontend.linear(attention, out_w, out_b)
        hidden_states = frontend.add(residual, attention)

        residual = hidden_states
        ln2_w = frontend.add_input(f"h{layer}_ln_2_weight", DType.F32, (hidden,))
        ln2_b = frontend.add_input(f"h{layer}_ln_2_bias", DType.F32, (hidden,))
        fc_w = frontend.add_input(
            f"h{layer}_mlp_c_fc_weight", DType.F32, (mlp_hidden, hidden)
        )
        fc_b = frontend.add_input(f"h{layer}_mlp_c_fc_bias", DType.F32, (mlp_hidden,))
        proj_w = frontend.add_input(
            f"h{layer}_mlp_c_proj_weight", DType.F32, (hidden, mlp_hidden)
        )
        proj_b = frontend.add_input(f"h{layer}_mlp_c_proj_bias", DType.F32, (hidden,))
        hidden_states = frontend.layer_norm(hidden_states, ln2_w, ln2_b)
        mlp = frontend.linear(hidden_states, fc_w, fc_b)
        mlp = frontend.gelu(mlp)
        mlp = frontend.linear(mlp, proj_w, proj_b)
        hidden_states = frontend.add(residual, mlp)

    hidden_states = frontend.layer_norm(hidden_states, final_ln_weight, final_ln_bias)
    logits = frontend.linear(hidden_states, lm_head)
    if output_logits:
        frontend.output(logits)
    else:
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

    def test_full_gpt2_graph_builds_and_validates(self):
        graph = build_full_gpt2_graph(seq=8)
        graph.validate()
        self.assertEqual(graph.outputs[0].shape, (1, 8))
        self.assertIn("masked_attention", {node.op.value for node in graph.nodes})
        self.assertIn("gelu", {node.op.value for node in graph.nodes})

    def test_multi_head_attention_mlir_builder(self):
        frontend = ManualFrontend("multi_head_attention")
        q = frontend.add_input("q", DType.F32, (1, 12, 8, 64))
        k = frontend.add_input("k", DType.F32, (1, 12, 8, 64))
        v = frontend.add_input("v", DType.F32, (1, 12, 8, 64))
        mask = frontend.add_input("mask", DType.F32, (1, 1, 8, 8))
        frontend.output(
            frontend.masked_attention(q, k, v, mask, scale=1.0 / (64.0**0.5))
        )
        mlir_text = MLIRBuilder().build(frontend.build())
        self.assertIn("mgpu.attention", mlir_text)
        self.assertIn("memref<1x12x8x64xf32>", mlir_text)


if __name__ == "__main__":
    unittest.main()
