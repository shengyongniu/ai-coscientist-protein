"""Reflection agent: a virtual peer reviewer scoring each hypothesis."""

from __future__ import annotations

from coscientist.agents.base import Agent
from coscientist.core.models import AgentKind, Hypothesis, Review


class ReflectionAgent(Agent):
    kind = AgentKind.REFLECTION

    def review(self, hypothesis: Hypothesis, round: int = 0) -> Review:
        cfg = self.ctx.config
        data = self._call_json(
            "reflection.system.j2",
            "reflection.user.j2",
            model="strong",
            temperature=cfg.reflection.temperature,
            domain=self.ctx.domain,
            goal=cfg.goal or self.ctx.session.goal,
            hypothesis=hypothesis,
        )

        def _clamp(x) -> float:
            try:
                return max(0.0, min(10.0, float(x)))
            except (TypeError, ValueError):
                return 5.0

        review = Review(
            hypothesis_id=hypothesis.id,
            correctness=_clamp(data.get("correctness", 5)),
            novelty=_clamp(data.get("novelty", 5)),
            testability=_clamp(data.get("testability", 5)),
            safety=_clamp(data.get("safety", 5)),
            critique=data.get("critique", ""),
            suggestions=data.get("suggestions", []) or [],
        )
        hypothesis.review_id = review.id
        self.ctx.event(
            self.kind, "review", round=round,
            message=f"Reviewed {hypothesis.title} (avg {review.aggregate})",
            hypothesis_id=hypothesis.id, aggregate=review.aggregate,
        )
        return review
