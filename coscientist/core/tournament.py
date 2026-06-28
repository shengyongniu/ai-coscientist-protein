"""Elo tournament: the 'tournament of ideas'.

Hypotheses accumulate Elo ratings from pairwise matches. A match winner is
decided by a combination of (a) an objective protein score when available and
(b) a simulated scientific debate adjudicated by the LLM. This module owns only
the rating math and pairing logic; the actual debate adjudication is supplied by
the Ranking agent as a callback.
"""

from __future__ import annotations

import itertools
import random

from coscientist.core.models import Hypothesis

K_FACTOR = 32.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expectation that A beats B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float, a_won: bool, k: float = K_FACTOR) -> tuple[float, float]:
    """Return updated (rating_a, rating_b) after a match."""
    ea = expected_score(rating_a, rating_b)
    sa = 1.0 if a_won else 0.0
    new_a = rating_a + k * (sa - ea)
    new_b = rating_b + k * ((1.0 - sa) - (1.0 - ea))
    return new_a, new_b


def apply_result(a: Hypothesis, b: Hypothesis, winner_id: str, k: float = K_FACTOR) -> None:
    """Mutate two hypotheses' Elo + win/loss counts given a match winner."""
    a_won = winner_id == a.id
    a.elo, b.elo = update_elo(a.elo, b.elo, a_won, k=k)
    if a_won:
        a.wins += 1
        b.losses += 1
    else:
        b.wins += 1
        a.losses += 1


def make_pairings(
    hypotheses: list[Hypothesis],
    max_matches: int,
    proximity_groups: list[list[str]] | None = None,
    rng: random.Random | None = None,
) -> list[tuple[str, str]]:
    """Choose informative pairings for this round.

    Strategy: prioritize pairs that are close in Elo (informative matches), and
    when proximity groups are provided, prefer cross-group pairs to surface
    genuinely different ideas rather than near-duplicates.
    """
    rng = rng or random.Random()
    active = [h for h in hypotheses if h.active]
    if len(active) < 2:
        return []

    # Rank by Elo and pair neighbors (Swiss-style) for informative matches.
    ranked = sorted(active, key=lambda h: h.elo, reverse=True)
    candidate_pairs: list[tuple[str, str]] = []
    for i in range(len(ranked) - 1):
        candidate_pairs.append((ranked[i].id, ranked[i + 1].id))

    # Add some random cross pairings for exploration.
    ids = [h.id for h in active]
    rng.shuffle(ids)
    for a, b in itertools.pairwise(ids):
        candidate_pairs.append((a, b))

    # Optionally bias toward cross-cluster pairs for diversity.
    if proximity_groups:
        group_of: dict[str, int] = {}
        for gi, grp in enumerate(proximity_groups):
            for hid in grp:
                group_of[hid] = gi
        candidate_pairs.sort(key=lambda p: 0 if group_of.get(p[0]) != group_of.get(p[1]) else 1)

    # Deduplicate (order-insensitive) and cap.
    seen: set[frozenset[str]] = set()
    pairings: list[tuple[str, str]] = []
    for a, b in candidate_pairs:
        if a == b:
            continue
        key = frozenset((a, b))
        if key in seen:
            continue
        seen.add(key)
        pairings.append((a, b))
        if len(pairings) >= max_matches:
            break
    return pairings


def leaderboard(hypotheses: list[Hypothesis], top_k: int | None = None) -> list[Hypothesis]:
    ranked = sorted(
        [h for h in hypotheses if h.active], key=lambda h: h.elo, reverse=True
    )
    return ranked[:top_k] if top_k else ranked
