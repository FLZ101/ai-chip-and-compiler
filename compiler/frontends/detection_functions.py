"""Traceable PyTorch detection-head primitives.

The full detection head is intentionally expressed with two wrapped functions.
``torch.fx.symbolic_trace`` records ``call_function`` nodes instead of tracing
into their Python bodies, which lets :class:`TorchFXFrontend` lower them
directly to the Model IR ``box_decode`` and ``nms`` operations.
"""

from __future__ import annotations

import torch


@torch.fx.wrap
def box_decode(boxes: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Decode anchor boxes using YOLO/SSD-style center-form deltas."""
    cx = boxes[:, 0] + deltas[:, 0] * boxes[:, 2]
    cy = boxes[:, 1] + deltas[:, 1] * boxes[:, 3]
    width = boxes[:, 2] * torch.exp(deltas[:, 2])
    height = boxes[:, 3] * torch.exp(deltas[:, 3])
    return torch.stack(
        [cx - 0.5 * width, cy - 0.5 * height, cx + 0.5 * width, cy + 0.5 * height],
        dim=1,
    )


@torch.fx.wrap
def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """Traceable keep-all placeholder for the exported detection head.

    The production FX frontend recognizes this wrapped call and lowers it to
    ``mgpu.nms``.  This body only needs to be traceable; the simulator kernel
    implements the actual overlap suppression semantics.
    """
    return torch.arange(scores.shape[0], dtype=torch.float32)
