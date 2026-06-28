"""Protein scoring tools used to ground the Elo tournament in objective metrics.

Layered backends behind one `Scorer` interface:
  - HeuristicScorer  : numpy-only, always available (CPU fallback / CI).
  - ESMScorer        : ESM2 pseudo-log-likelihood (needs torch + transformers).
  - PredictorScorer  : your fine-tuned ESM2 regression head checkpoint.
"""

from coscientist.protein.factory import get_scorer
from coscientist.protein.scorer import Scorer, ScoreResult

__all__ = ["Scorer", "ScoreResult", "get_scorer"]
