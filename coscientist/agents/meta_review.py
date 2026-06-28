"""Meta-review agent: synthesizes the final research overview."""

from __future__ import annotations

from coscientist.agents.base import Agent
from coscientist.core.models import AgentKind, Hypothesis, ResearchOverview, Review


class MetaReviewAgent(Agent):
    kind = AgentKind.META_REVIEW

    def synthesize(
        self,
        top: list[Hypothesis],
        reviews: list[Review],
        rounds: int,
    ) -> ResearchOverview:
        cfg = self.ctx.config
        goal = cfg.goal or self.ctx.session.goal
        review_summary = self._summarize_reviews(reviews)
        self.ctx.event(self.kind, "start", message="Synthesizing research overview")
        data = self._call_json(
            "meta_review.system.j2",
            "meta_review.user.j2",
            model="strong",
            temperature=0.5,
            max_tokens=2500,
            domain=self.ctx.domain,
            goal=goal,
            top=top,
            rounds=rounds,
            review_summary=review_summary,
        )
        overview = ResearchOverview(
            session_id=self.ctx.session.id,
            goal=goal,
            summary=data.get("summary", ""),
            top_hypotheses=top,
            recommended_experiments=data.get("recommended_experiments", []) or [],
            open_questions=data.get("open_questions", []) or [],
        )
        overview.markdown = self.to_markdown(overview)
        self.ctx.event(self.kind, "done", message="Research overview ready")
        return overview

    @staticmethod
    def _summarize_reviews(reviews: list[Review]) -> str:
        if not reviews:
            return "No reviews available."
        n = len(reviews)
        avg = lambda f: round(sum(getattr(r, f) for r in reviews) / n, 2)  # noqa: E731
        return (
            f"{n} reviews. Mean scores — correctness {avg('correctness')}, "
            f"novelty {avg('novelty')}, testability {avg('testability')}, "
            f"safety {avg('safety')}."
        )

    @staticmethod
    def to_markdown(o: ResearchOverview) -> str:
        lines = [
            "# Research Overview",
            "",
            f"**Goal:** {o.goal}",
            "",
            "## Summary",
            o.summary,
            "",
            "## Top Hypotheses",
        ]
        for i, h in enumerate(o.top_hypotheses, 1):
            lines.append(f"### {i}. {h.title}  (Elo {h.elo:.0f}, {h.wins}W/{h.losses}L)")
            if h.protein_score is not None:
                lines.append(f"*Objective score:* {h.protein_score:.4f} ({h.scorer_name})")
            lines.append("")
            lines.append(h.summary)
            if h.rationale:
                lines.append("")
                lines.append(f"*Rationale:* {h.rationale}")
            if h.sequence:
                lines.append("")
                lines.append(f"```\n{h.sequence}\n```")
            if h.experiments:
                lines.append("")
                lines.append("*Proposed experiments:*")
                lines.extend(f"- {e}" for e in h.experiments)
            lines.append("")
        lines += ["## Recommended Experiments"]
        lines += [f"- {e}" for e in o.recommended_experiments] or ["- (none)"]
        lines += ["", "## Open Questions"]
        lines += [f"- {q}" for q in o.open_questions] or ["- (none)"]
        return "\n".join(lines)
