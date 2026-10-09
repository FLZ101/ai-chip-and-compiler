"""Pretrained CNN import/canonicalization test.

Phase 6.6 targets real pretrained models.  The current compiler can import and
canonicalize a full ResNet18 module; running the complete model in the
functional simulator is intentionally gated behind the hardware-scaling work
documented in ``docs/next-stage-steps.md``.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import torch
from torchvision.models import ResNet18_Weights, resnet18

from compiler.mlir import (
    MGpuIsaBundle,
    MLIRCanonicalizer,
    TorchMLIRImporter,
    run_model_pipeline,
    run_model_pipeline_with_memory_plan,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_RESNET18_WEIGHTS = (
    _PROJECT_ROOT / ".cache" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
)
_HOME_RESNET18_WEIGHTS = (
    Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
)
RESNET18_WEIGHTS = (
    _PROJECT_RESNET18_WEIGHTS
    if _PROJECT_RESNET18_WEIGHTS.exists()
    else _HOME_RESNET18_WEIGHTS
)


@unittest.skipUnless(
    RESNET18_WEIGHTS.exists() or os.environ.get("MGPU_DOWNLOAD_MODELS") == "1",
    "download real models with scripts/download_real_models.py or set MGPU_DOWNLOAD_MODELS=1",
)
class TestPretrainedCNNImport(unittest.TestCase):
    def test_pretrained_resnet18_imports_and_canonicalizes(self):
        model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
        example = torch.zeros(1, 3, 224, 224)

        imported = TorchMLIRImporter().import_module(
            model,
            example,
            func_name="resnet18",
        )
        canonicalized = MLIRCanonicalizer().canonicalize(imported)

        self.assertEqual(canonicalized.source, "fx")
        self.assertEqual(canonicalized.target, "linalg-on-tensors")
        self.assertIn("linalg.conv_2d_nchw_fchw", canonicalized.text)
        self.assertIn("linalg.generic", canonicalized.text)
        self.assertNotIn("torch.aten", canonicalized.text)
        self.assertTrue(canonicalized.mlir_module.operation.verify())

    def test_full_resnet18_lowers_to_isa_programs(self):
        """Phase A of ``docs/resnet18-run-plan.md``.

        The full canonicalized module should survive one-shot bufferization,
        map supported linalg compute to ``mgpu.launch``, and emit one or more
        ISA programs in program order.  This is a structural lowering test;
        full numerical execution is still gated behind later ResNet18 phases.
        """
        model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
        example = torch.zeros(1, 3, 224, 224)

        imported = TorchMLIRImporter().import_module(
            model,
            example,
            func_name="resnet18",
        )
        canonicalized = MLIRCanonicalizer().canonicalize(imported)

        bundle = run_model_pipeline(canonicalized.text)

        self.assertIsInstance(bundle, MGpuIsaBundle)
        self.assertGreaterEqual(len(bundle.programs), 1)
        self.assertTrue(all(program.code for program in bundle.programs))
        self.assertTrue(all(program.grid for program in bundle.programs))
        self.assertTrue(all(program.block for program in bundle.programs))

    def test_full_resnet18_memory_plan_is_deterministic_and_fits(self):
        """Phase C of ``docs/resnet18-run-plan.md``.

        The full module is bufferized, all static memref buffers are planned
        with lifetime-aware reuse, and the resulting ``mgpu.memory_layout`` is
        injected before lowering.  The plan must fit the configured 64 MiB
        global-memory budget and be identical across repeated runs.
        """
        model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
        example = torch.zeros(1, 3, 224, 224)
        imported = TorchMLIRImporter().import_module(
            model,
            example,
            func_name="resnet18",
        )
        canonicalized = MLIRCanonicalizer().canonicalize(imported)

        first = run_model_pipeline_with_memory_plan(canonicalized.text)
        second = run_model_pipeline_with_memory_plan(canonicalized.text)

        self.assertIsNotNone(first.memory_plan)
        self.assertIsNotNone(second.memory_plan)
        self.assertLessEqual(first.memory_plan.peak_bytes, 64 * 1024 * 1024)
        self.assertEqual(
            first.memory_plan.peak_bytes,
            second.memory_plan.peak_bytes,
        )
        self.assertEqual(
            first.memory_plan.total_bytes,
            second.memory_plan.total_bytes,
        )
        self.assertEqual(len(first.programs), len(second.programs))

    def test_full_resnet18_final_linear_lowers_to_isa(self):
        """Phase D structural gate for the final classifier.

        The full model pipeline must emit a bounded ``linear_loop`` ISA program
        for ResNet18's ``512 -> 1000`` classifier.
        """
        model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
        example = torch.zeros(1, 3, 224, 224)
        imported = TorchMLIRImporter().import_module(
            model,
            example,
            func_name="resnet18",
        )
        canonicalized = MLIRCanonicalizer().canonicalize(imported)
        bundle = run_model_pipeline_with_memory_plan(canonicalized.text)

        self.assertTrue(
            any(
                "mat_loop" in program.code and program.grid == (32, 1, 1)
                for program in bundle.programs
            )
        )


if __name__ == "__main__":
    unittest.main()
