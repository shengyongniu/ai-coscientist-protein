"""Scorer factory with graceful degradation.

Resolution: requested scorer -> fall back to heuristic if heavy deps or a
checkpoint are unavailable. This guarantees the engine always has a working
scorer, keeping the repo runnable for everyone.
"""

from __future__ import annotations

import os
import warnings

from coscientist.protein.scorer import HeuristicScorer, Scorer


def get_scorer(name: str | None = None, checkpoint: str | None = None) -> Scorer:
    name = (name or os.getenv("COSCIENTIST_SCORER", "heuristic")).lower()
    checkpoint = checkpoint or os.getenv("COSCIENTIST_PREDICTOR_CKPT", "")

    if name == "heuristic":
        return HeuristicScorer()

    if name == "predictor":
        if checkpoint and os.path.exists(checkpoint):
            try:
                from coscientist.protein.predictor import PredictorScorer

                return PredictorScorer(checkpoint)
            except Exception as e:  # pragma: no cover - env dependent
                warnings.warn(f"PredictorScorer unavailable ({e}); falling back to esm.", stacklevel=2)
                name = "esm"
        else:
            warnings.warn("No predictor checkpoint found; falling back to esm.", stacklevel=2)
            name = "esm"

    if name == "esm":
        try:
            from coscientist.protein.esm_scorer import ESMScorer

            return ESMScorer()
        except Exception as e:  # pragma: no cover - env dependent
            warnings.warn(
                f"ESMScorer unavailable ({e}); falling back to heuristic.", stacklevel=2
            )
            return HeuristicScorer()

    raise ValueError(f"Unknown scorer: {name}")
