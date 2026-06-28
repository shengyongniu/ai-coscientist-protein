"""Ranking agent: runs the Elo 'tournament of ideas'.

Each match blends two signals into a win probability:
  - the objective protein score difference (when both have scores), and
  - a simulated scientific debate adjudicated by the LLM.
``score_weight`` in config controls the mix. Elo ratings are then updated.
"""

from __future__ import annotations

import random

from coscientist.agents.base import Agent
from coscientist.core import tournament
from coscientist.core.models import AgentKind, DebateResult, Hypothesis


class RankingAgent(Agent):
    kind = AgentKind.RANKING

    def run_tournament(
        self,
        hypotheses: list[Hypothesis],
        round: int = 0,
        proximity_groups: list[list[str]] | None = None,
        rng: random.Random | None = None,
    ) -> list[DebateResult]:
        cfg = self.ctx.config
        rng = rng or random.Random(round)
        by_id = {h.id: h for h in hypotheses}
        pairings = tournament.make_pairings(
            hypotheses, cfg.ranking.matches_per_round, proximity_groups, rng
        )
        self.ctx.event(
            self.kind, "start", round=round, message=f"Tournament: {len(pairings)} matches"
        )
        results: list[DebateResult] = []
        for a_id, b_id in pairings:
            a, b = by_id[a_id], by_id[b_id]
            winner_id = self._decide(a, b, cfg.ranking.score_weight, rng)
            reasoning = self._last_reasoning
            tournament.apply_result(a, b, winner_id)
            res = DebateResult(
                round=round,
                hypothesis_a=a_id,
                hypothesis_b=b_id,
                winner=winner_id,
                reasoning=reasoning,
            )
            results.append(res)
            self.ctx.event(
                self.kind, "match", round=round,
                message=f"{by_id[winner_id].title} won",
                a=a_id, b=b_id, winner=winner_id,
            )

        board = tournament.leaderboard(hypotheses, top_k=10)
        self.ctx.event(
            self.kind, "leaderboard", round=round, message="Updated leaderboard",
            leaderboard=[
                {"id": h.id, "title": h.title, "elo": round_(h.elo),
                 "score": h.protein_score, "wins": h.wins, "losses": h.losses}
                for h in board
            ],
        )
        return results

    _last_reasoning: str = ""

    def _decide(self, a: Hypothesis, b: Hypothesis, score_weight: float, rng: random.Random) -> str:
        """Return the winning hypothesis id by blending score + debate."""
        # Score-based win probability for A.
        p_score = 0.5
        if a.protein_score is not None and b.protein_score is not None:
            diff = a.protein_score - b.protein_score
            p_score = 1.0 / (1.0 + 10.0 ** (-diff))  # logistic on score diff

        # Debate-based decision from the LLM.
        p_debate = 0.5
        self._last_reasoning = ""
        try:
            data = self._call_json(
                "ranking.system.j2",
                "ranking.user.j2",
                model="fast",
                temperature=self.ctx.config.ranking.temperature,
                max_tokens=600,
                domain=self.ctx.domain,
                goal=self.ctx.config.goal or self.ctx.session.goal,
                a=a,
                b=b,
            )
            winner = str(data.get("winner", "A")).strip().upper()
            self._last_reasoning = data.get("reasoning", "")
            p_debate = 1.0 if winner == "A" else 0.0
        except ValueError:
            p_debate = 0.5

        p_a = score_weight * p_score + (1.0 - score_weight) * p_debate
        return a.id if rng.random() < p_a else b.id


def round_(x: float) -> float:
    return round(x, 1)
