"""Real GPT-2 software execution gate.

This opt-in test runs the full 12-layer GPT-2-small forward through the
software operator runner and compares greedy token IDs with PyTorch.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
import torch

from tests.model_e2e.test_real_llm_graph import build_full_gpt2_graph
from tests.operator_e2e.helpers import make_device, run_software_model_e2e

GPT2_CACHE = (
    Path(__file__).resolve().parents[2] / ".cache" / "gpt2-small" / "pytorch_model.bin"
)


def _gpt2_inputs(model, input_ids, seq):
    state = model.state_dict()
    wpe = state["transformer.wpe.weight"].detach().numpy()

    inputs = {
        "input_ids": input_ids.detach().numpy(),
        "token_embed": state["transformer.wte.weight"].detach().numpy(),
        "pos_embed": wpe[:seq][None, :, :],
        "causal_mask": np.tril(np.ones((seq, seq), dtype=np.float32)).reshape(
            1, 1, seq, seq
        ),
        "final_ln_weight": state["transformer.ln_f.weight"].detach().numpy(),
        "final_ln_bias": state["transformer.ln_f.bias"].detach().numpy(),
        "lm_head": model.lm_head.weight.detach().numpy(),
    }

    for layer in range(12):
        block = model.transformer.h[layer]
        c_attn_weight = state[f"transformer.h.{layer}.attn.c_attn.weight"]
        c_attn_bias = state[f"transformer.h.{layer}.attn.c_attn.bias"]
        hidden = block.ln_1.weight.shape[0]
        c_attn_weight = c_attn_weight.detach().numpy()
        c_attn_bias = c_attn_bias.detach().numpy()

        inputs[f"h{layer}_ln_1_weight"] = block.ln_1.weight.detach().numpy()
        inputs[f"h{layer}_ln_1_bias"] = block.ln_1.bias.detach().numpy()
        inputs[f"h{layer}_q_weight"] = c_attn_weight[:, 0:hidden].T
        inputs[f"h{layer}_q_bias"] = c_attn_bias[0:hidden]
        inputs[f"h{layer}_k_weight"] = c_attn_weight[:, hidden : 2 * hidden].T
        inputs[f"h{layer}_k_bias"] = c_attn_bias[hidden : 2 * hidden]
        inputs[f"h{layer}_v_weight"] = c_attn_weight[:, 2 * hidden : 3 * hidden].T
        inputs[f"h{layer}_v_bias"] = c_attn_bias[2 * hidden : 3 * hidden]
        inputs[f"h{layer}_out_weight"] = block.attn.c_proj.weight.detach().numpy().T
        inputs[f"h{layer}_out_bias"] = block.attn.c_proj.bias.detach().numpy()
        inputs[f"h{layer}_ln_2_weight"] = block.ln_2.weight.detach().numpy()
        inputs[f"h{layer}_ln_2_bias"] = block.ln_2.bias.detach().numpy()
        inputs[f"h{layer}_mlp_c_fc_weight"] = block.mlp.c_fc.weight.detach().numpy().T
        inputs[f"h{layer}_mlp_c_fc_bias"] = block.mlp.c_fc.bias.detach().numpy()
        inputs[f"h{layer}_mlp_c_proj_weight"] = (
            block.mlp.c_proj.weight.detach().numpy().T
        )
        inputs[f"h{layer}_mlp_c_proj_bias"] = block.mlp.c_proj.bias.detach().numpy()

    return inputs


@unittest.skipUnless(
    (GPT2_CACHE.exists() or os.environ.get("MGPU_DOWNLOAD_MODELS") == "1")
    and os.environ.get("MGPU_RUN_GPT2_SOFTWARE") == "1",
    "set MGPU_RUN_GPT2_SOFTWARE=1 to run the full GPT-2 software gate",
)
class TestRealLLMDecode(unittest.TestCase):
    def test_full_gpt2_software_forward_matches_pytorch(self):
        from transformers import GPT2LMHeadModel

        model = GPT2LMHeadModel.from_pretrained(
            str(GPT2_CACHE.parent),
            local_files_only=True,
        ).eval()
        seq = 8
        input_ids = torch.zeros((1, seq), dtype=torch.long)
        inputs = _gpt2_inputs(model, input_ids, seq)

        with torch.no_grad():
            logits = model(input_ids, use_cache=False).logits
            expected_ids = logits.argmax(dim=-1).detach().numpy()

        graph = build_full_gpt2_graph(
            seq=seq,
            vocab=50257,
            hidden=768,
            heads=12,
            layers=12,
            output_logits=True,
        )
        result = run_software_model_e2e(
            make_device(global_size=1 << 30, sm_count=16),
            graph,
            inputs,
            progress=True,
            use_native=False,
        )
        actual_logits = np.asarray(result.outputs[0], dtype=np.float32)
        np.testing.assert_allclose(
            actual_logits,
            logits.detach().numpy(),
            rtol=1e-2,
            atol=1e-2,
        )
        actual_ids = actual_logits.argmax(axis=-1)
        np.testing.assert_array_equal(actual_ids, expected_ids)


if __name__ == "__main__":
    unittest.main()
