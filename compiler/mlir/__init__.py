"""MLIR text generation, import, and pass definitions for the minimal GPU target."""

from .canonicalizer import (
    CanonicalizedMLIRResult,
    MLIRCanonicalizationError,
    MLIRCanonicalizer,
)
from .importer import (
    ONNXImportError,
    ONNXImportResult,
    ONNXImporter,
    TorchMLIRImportError,
    TorchMLIRImporter,
    TorchMLIRImportResult,
)
from .mlir_builder import MLIRBuilder, MLIRBuilderError
from .mgpu_runner import (
    DEFAULT_MODEL_PIPELINE,
    MGpuIsaBundle,
    MGpuIsaResult,
    MGpuOptError,
    enqueue_isa_bundle,
    lower_with_mgpu_opt,
    run_mgpu_pipeline,
    run_model_pipeline,
    run_model_pipeline_with_memory_plan,
)
from .mgpu_to_isa import MGpuToIsaError, lower_mgpu_to_isa
from .passes import DEFAULT_PASS_PIPELINE, Pass, PassPipeline
from .pipeline import DEFAULT_MGPU_OPT_PIPELINE

__all__ = [
    "CanonicalizedMLIRResult",
    "DEFAULT_PASS_PIPELINE",
    "DEFAULT_MODEL_PIPELINE",
    "DEFAULT_MGPU_OPT_PIPELINE",
    "MLIRCanonicalizationError",
    "MLIRCanonicalizer",
    "MLIRBuilder",
    "MLIRBuilderError",
    "MGpuToIsaError",
    "MGpuIsaBundle",
    "MGpuIsaResult",
    "MGpuOptError",
    "ONNXImportError",
    "ONNXImportResult",
    "ONNXImporter",
    "Pass",
    "PassPipeline",
    "TorchMLIRImportError",
    "TorchMLIRImporter",
    "TorchMLIRImportResult",
    "enqueue_isa_bundle",
    "lower_mgpu_to_isa",
    "lower_with_mgpu_opt",
    "run_mgpu_pipeline",
    "run_model_pipeline",
    "run_model_pipeline_with_memory_plan",
]
