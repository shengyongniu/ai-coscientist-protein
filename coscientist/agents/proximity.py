"""Proximity agent: embeds and clusters hypotheses for dedup + diverse pairings.

To keep the core engine dependency-light (numpy only), embeddings are computed
locally: k-mer frequency vectors for protein sequences and hashed bag-of-words
for free text. This is enough to detect near-duplicates and group similar ideas;
a heavier sentence-transformer could be swapped in behind the same interface.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from coscientist.agents.base import Agent
from coscientist.core.models import AgentKind, Hypothesis

_AA = "ACDEFGHIKLMNPQRSTVWY"
_DIM = 256


def _kmer_vector(seq: str, k: int = 3) -> np.ndarray:
    vec = np.zeros(_DIM, dtype=np.float32)
    seq = "".join(c for c in seq.upper() if c in _AA)
    if len(seq) < k:
        return vec
    for i in range(len(seq) - k + 1):
        h = int(hashlib.md5(seq[i : i + k].encode()).hexdigest(), 16) % _DIM
        vec[h] += 1.0
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def _text_vector(text: str) -> np.ndarray:
    vec = np.zeros(_DIM, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % _DIM
        vec[h] += 1.0
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def embed(h: Hypothesis) -> np.ndarray:
    if h.sequence:
        return _kmer_vector(h.sequence)
    return _text_vector(f"{h.title} {h.summary} {h.rationale}")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # vectors are already normalized


class ProximityAgent(Agent):
    kind = AgentKind.PROXIMITY

    def deduplicate(
        self, hypotheses: list[Hypothesis], threshold: float, round: int = 0
    ) -> tuple[list[Hypothesis], list[list[str]]]:
        """Mark near-duplicates inactive (keep the higher-Elo one) and return
        (active_hypotheses, proximity_groups).
        """
        active = [h for h in hypotheses if h.active]
        embs = {h.id: embed(h) for h in active}
        # Greedy dedup: process by Elo desc, drop later ones too close to a kept one.
        kept: list[Hypothesis] = []
        for h in sorted(active, key=lambda x: x.elo, reverse=True):
            dup_of = None
            for k in kept:
                if cosine(embs[h.id], embs[k.id]) >= threshold:
                    dup_of = k
                    break
            if dup_of is None:
                kept.append(h)
            else:
                h.active = False
                self.ctx.event(
                    self.kind, "log", round=round,
                    message=f"Deduplicated '{h.title}' (~ '{dup_of.title}')",
                    hypothesis_id=h.id,
                )

        # Cluster the kept set by simple connected components at a looser threshold.
        groups = self._cluster(kept, embs, max(0.0, threshold - 0.15))
        self.ctx.event(
            self.kind, "log", round=round,
            message=f"{len(kept)} unique hypotheses in {len(groups)} clusters",
            clusters=len(groups), unique=len(kept),
        )
        return kept, groups

    @staticmethod
    def _cluster(hyps: list[Hypothesis], embs: dict, thresh: float) -> list[list[str]]:
        ids = [h.id for h in hyps]
        parent = {i: i for i in ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if cosine(embs[ids[i]], embs[ids[j]]) >= thresh:
                    union(ids[i], ids[j])
        groups: dict[str, list[str]] = {}
        for i in ids:
            groups.setdefault(find(i), []).append(i)
        return list(groups.values())
