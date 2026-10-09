#!/usr/bin/env python3
"""Download real pretrained CNN weights used by Phase 6.6.

The simulator/compiler stack can import these models into MLIR.  Full
simulator execution is gated on the hardware-scaling work described in
``docs/next-stage-steps.md``, but this script keeps the pretrained checkpoints
available for the importer and canonicalizer tests.
"""

from __future__ import annotations

import argparse

from torchvision.models import ResNet18_Weights, resnet18


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("resnet18",),
        default="resnet18",
        help="pretrained model to download",
    )
    args = parser.parse_args()

    if args.model == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        print(
            f"downloaded {args.model} with {sum(p.numel() for p in model.parameters())} parameters"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
