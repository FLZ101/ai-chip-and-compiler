"""Traceable PyTorch decoder-attention primitives.

``causal_mask`` and ``masked_attention`` are wrapped FX functions.  The
frontend lowers their call nodes directly to ``mgpu.causal_mask`` and
``mgpu.attention`` instead of tracing the reference implementations, which
keeps the exported graph compact and avoids duplicating the simulator kernels.
"""

from __future__ import annotations

import torch


@torch.fx.wrap
def causal_mask(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Traceable causal-mask placeholder.

    The actual lower-triangular mask is produced by ``mgpu.causal_mask``.
    This body only needs to be traceable.
    """
    return torch.zeros_like(query)


@torch.fx.wrap
def masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Traceable masked scaled-dot-product-attention placeholder."""
    return torch.zeros_like(q)
