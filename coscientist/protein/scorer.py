"""Scorer interface + a dependency-free heuristic scorer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoreResult:
    score: float  # higher is better
    detail: dict[str, Any] = field(default_factory=dict)


class Scorer(ABC):
    name: str = "base"

    @abstractmethod
    def score(self, sequence: str) -> ScoreResult: ...

    def score_many(self, sequences: list[str]) -> list[ScoreResult]:
        return [self.score(s) for s in sequences]


# Kyte-Doolittle hydropathy; used as a crude developability proxy.
_HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
_AA = set("ACDEFGHIKLMNPQRSTVWY")


class HeuristicScorer(Scorer):
    """A fast, transparent proxy score.

    Combines: composition validity, a mild penalty for extreme average
    hydrophobicity (aggregation proxy), and a penalty for long hydrophobic runs.
    It is NOT biophysically accurate; it exists so the engine and tournament run
    everywhere without torch. Higher is better, roughly in [0, 1].
    """

    name = "heuristic"

    def score(self, sequence: str) -> ScoreResult:
        seq = "".join(c for c in (sequence or "").upper() if c.isalpha())
        if not seq:
            return ScoreResult(score=0.0, detail={"reason": "empty"})

        valid = sum(1 for c in seq if c in _AA) / len(seq)
        mean_hydro = sum(_HYDROPATHY.get(c, 0.0) for c in seq) / len(seq)
        # Penalize deviation from a mildly hydrophilic ideal (~ -0.4).
        hydro_pen = abs(mean_hydro + 0.4) / 4.5

        # Longest hydrophobic run.
        run = best = 0
        for c in seq:
            if _HYDROPATHY.get(c, 0.0) > 1.5:
                run += 1
                best = max(best, run)
            else:
                run = 0
        run_pen = min(best / 10.0, 1.0)

        score = max(0.0, valid - 0.5 * hydro_pen - 0.3 * run_pen)
        return ScoreResult(
            score=round(score, 4),
            detail={
                "validity": round(valid, 3),
                "mean_hydropathy": round(mean_hydro, 3),
                "max_hydrophobic_run": best,
                "length": len(seq),
            },
        )
