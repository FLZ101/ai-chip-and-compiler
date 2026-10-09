"""Minimal autotuning hooks for the simulated GPU runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class TuningConfig:
    """A candidate configuration for a generated kernel."""

    tile_m: int = 16
    tile_n: int = 16
    tile_k: int = 16
    mma_layout: str = "default"
    shared_padding: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
            "mma_layout": self.mma_layout,
            "shared_padding": self.shared_padding,
        }


@dataclass(frozen=True)
class TuningResult:
    """Best result from :func:`autotune`."""

    config: TuningConfig
    score: float
    outputs: Any


def autotune(
    candidates: Iterable[TuningConfig],
    runner: Callable[[TuningConfig], tuple[float, Any]],
) -> TuningResult:
    """Evaluate candidates deterministically and return the best result.

    ``runner`` returns ``(score, outputs)`` for one configuration.  Smaller
    scores are better.  The first candidate wins ties, so repeated runs are
    deterministic.  Numerical outputs must not change across candidates.
    """
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("autotune requires at least one candidate")

    best_score: float | None = None
    best_config = candidate_list[0]
    best_outputs: Any = None

    for config in candidate_list:
        score, outputs = runner(config)
        if best_score is None or score < best_score:
            best_score = score
            best_config = config
            best_outputs = outputs

    if best_outputs is None:
        raise RuntimeError("autotune runner returned no result")
    return TuningResult(
        config=best_config, score=float(best_score), outputs=best_outputs
    )
