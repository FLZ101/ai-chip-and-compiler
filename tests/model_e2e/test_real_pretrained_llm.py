"""Pretrained LLM import/canonicalization test.

Phase 6.7 targets small pretrained decoder models.  The current compiler can
import and canonicalize a pretrained GPT-2 transformer block.  Full greedy
decode execution remains gated on the hardware-scaling work in
``docs/next-stage-steps.md``.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from compiler.mlir import MLIRCanonicalizer, TorchMLIRImporter

GPT2_CACHE = (
    Path(__file__).resolve().parents[2] / ".cache" / "gpt2-small" / "pytorch_model.bin"
)


class GPT2BlockWrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden_states):
        return self.block(hidden_states, use_cache=False)


@unittest.skipUnless(
    GPT2_CACHE.exists() or os.environ.get("MGPU_DOWNLOAD_MODELS") == "1",
    "download GPT-2 small with scripts/download_real_models.py or set MGPU_DOWNLOAD_MODELS=1",
)
class TestPretrainedLLMImport(unittest.TestCase):
    def test_pretrained_gpt2_block_imports_and_canonicalizes(self):
        from transformers import GPT2LMHeadModel

        model = GPT2LMHeadModel.from_pretrained(
            str(GPT2_CACHE.parent),
            local_files_only=True,
        ).eval()
        wrapper = GPT2BlockWrapper(model.transformer.h[0]).eval()
        example = torch.zeros(1, 8, 768)

        imported = TorchMLIRImporter().import_module(
            wrapper,
            example,
            func_name="gpt2_block",
        )
        canonicalized = MLIRCanonicalizer().canonicalize(imported)

        self.assertEqual(canonicalized.source, "fx")
        self.assertEqual(canonicalized.target, "linalg-on-tensors")
        self.assertIn("linalg.matmul", canonicalized.text)
        self.assertIn("linalg.generic", canonicalized.text)
        self.assertNotIn("torch.aten", canonicalized.text)
        self.assertTrue(canonicalized.mlir_module.operation.verify())


if __name__ == "__main__":
    unittest.main()
