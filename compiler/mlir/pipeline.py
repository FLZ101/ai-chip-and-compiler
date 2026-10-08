"""Concrete ``mgpu-opt`` pipelines used by the Python driver."""

from __future__ import annotations

# Phase 0 uses only the target backend pass.  The full production pipeline
# will be assembled from the registered C++ passes once each phase is
# implemented.
DEFAULT_MGPU_OPT_PIPELINE = "builtin.module(func.func(mgpu-to-isa))"


def mgpu_opt_pipeline_text() -> str:
    """Return the command-line ``--pass-pipeline`` value for the current phase."""
    return DEFAULT_MGPU_OPT_PIPELINE
