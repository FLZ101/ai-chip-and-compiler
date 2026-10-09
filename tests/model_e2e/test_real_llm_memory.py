"""GPT-2-small memory planning gate.

Step 6 of ``docs/gpt2-small-run-plan.md`` plans the static GPT-2 weight and
activation buffers without requiring the full torch-mlir module to lower to
ISA.  The full module currently contains ``tm_tensor.scan``, so this test uses
the same lifetime-aware planner on explicit GPT-2-sized buffers.
"""

from __future__ import annotations

import unittest

from runtime.memory_plan import PlannedBuffer, plan_memory


def build_gpt2_memory_plan(
    *,
    vocab: int = 50257,
    hidden: int = 768,
    heads: int = 12,
    layers: int = 12,
    seq: int = 8,
    alignment: int = 4096,
):
    head_dim = hidden // heads
    buffers: list[PlannedBuffer] = []

    def add(name: str, numel: int, live_from: int = 0, live_until: int | None = None):
        buffers.append(
            PlannedBuffer(
                name=name,
                numel=numel,
                dtype="f32",
                live_from=live_from,
                live_until=live_until,
            )
        )

    add("token_embed", vocab * hidden)
    add("pos_embed", seq * hidden)

    for layer in range(layers):
        add(f"h{layer}_q_weight", hidden * hidden)
        add(f"h{layer}_q_bias", hidden)
        add(f"h{layer}_k_weight", hidden * hidden)
        add(f"h{layer}_k_bias", hidden)
        add(f"h{layer}_v_weight", hidden * hidden)
        add(f"h{layer}_v_bias", hidden)
        add(f"h{layer}_out_weight", hidden * hidden)
        add(f"h{layer}_out_bias", hidden)
        add(f"h{layer}_ln_1_weight", hidden)
        add(f"h{layer}_ln_1_bias", hidden)
        add(f"h{layer}_ln_2_weight", hidden)
        add(f"h{layer}_ln_2_bias", hidden)
        add(f"h{layer}_mlp_c_fc_weight", 4 * hidden * hidden)
        add(f"h{layer}_mlp_c_fc_bias", 4 * hidden)
        add(f"h{layer}_mlp_c_proj_weight", hidden * 4 * hidden)
        add(f"h{layer}_mlp_c_proj_bias", hidden)

    add("final_ln_weight", hidden)
    add("final_ln_bias", hidden)
    add("lm_head", vocab * hidden)

    for layer in range(layers):
        add(f"activation_{layer}", seq * hidden, live_from=layer, live_until=layer + 1)

    add("logits", seq * vocab, live_from=layers, live_until=layers + 1)
    add(
        "kv_cache",
        2 * heads * seq * head_dim,
        live_from=layers,
        live_until=layers + 1,
    )

    return plan_memory(buffers, alignment=alignment)


class TestRealLLMMemory(unittest.TestCase):
    def test_gpt2_memory_plan_is_deterministic_and_fits(self):
        first = build_gpt2_memory_plan()
        second = build_gpt2_memory_plan()

        self.assertLessEqual(first.peak_bytes, 1 << 30)
        self.assertEqual(first.peak_bytes, second.peak_bytes)
        self.assertEqual(first.total_bytes, second.total_bytes)
        self.assertGreater(first.peak_bytes, 0)
        self.assertGreater(first.total_bytes, first.peak_bytes // 2)
        self.assertIn("token_embed", first.allocations)
        self.assertIn("lm_head", first.allocations)
        self.assertIn("kv_cache", first.allocations)


if __name__ == "__main__":
    unittest.main()
