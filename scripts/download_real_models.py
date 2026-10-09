#!/usr/bin/env python3
"""Download real pretrained CNN weights used by Phase 6.6.

The simulator/compiler stack can import these models into MLIR.  Full
simulator execution is gated on the hardware-scaling work described in
``docs/next-stage-steps.md``, but this script keeps the pretrained checkpoints
available for the importer and canonicalizer tests.
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

from torchvision.models import ResNet18_Weights, resnet18

GPT2_CONFIG_URL = "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json"
GPT2_WEIGHTS_URL = (
    "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin"
)

PROJECT_CACHE = Path(__file__).resolve().parents[1] / ".cache"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("resnet18", "gpt2-small", "all"),
        default="all",
        help="pretrained model to download",
    )
    args = parser.parse_args()

    if args.model in {"resnet18", "all"}:
        torch_home = PROJECT_CACHE / "torch"
        torch_home.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(torch_home)
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        print(
            f"downloaded {args.model} with {sum(p.numel() for p in model.parameters())} parameters"
        )
    if args.model in {"gpt2-small", "all"}:
        target = PROJECT_CACHE / "gpt2-small"
        target.mkdir(parents=True, exist_ok=True)
        for url, name in (
            (GPT2_CONFIG_URL, "config.json"),
            (GPT2_WEIGHTS_URL, "pytorch_model.bin"),
        ):
            destination = target / name
            if destination.exists():
                print(f"already cached {name}")
                continue
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, destination)
        print(f"gpt2-small cached in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
