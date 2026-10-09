"""Traceable PyTorch KV-cache primitives."""

from __future__ import annotations

import torch


@torch.fx.wrap
def kv_cache_append(cache: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Traceable KV-cache append placeholder.

    ``mgpu.kv_cache_append`` implements the simulator write semantics; this
    body only needs to be traceable.
    """
    return torch.zeros_like(cache)


@torch.fx.wrap
def kv_cache_gather(cache: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Traceable KV-cache gather placeholder."""
    return torch.zeros_like(cache)
