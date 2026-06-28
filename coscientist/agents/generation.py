"""Generation agent: proposes novel, diverse hypotheses for the research goal."""

from __future__ import annotations

from coscientist.agents.base import Agent
from coscientist.core.models import AgentKind, Hypothesis


class GenerationAgent(Agent):
    kind = AgentKind.GENERATION

    def generate(
        self,
        count: int,
        round: int = 0,
        feedback: str = "",
        existing_titles: list[str] | None = None,
    ) -> list[Hypothesis]:
        cfg = self.ctx.config
        self.ctx.event(self.kind, "start", round=round, message=f"Generating {count} hypotheses")
        data = self._call_json(
            "generation.system.j2",
            "generation.user.j2",
            model="strong",
            temperature=cfg.generation.temperature,
            domain=self.ctx.domain,
            goal=cfg.goal or self.ctx.session.goal,
            protein_mode=cfg.protein_mode,
            seed_sequence=cfg.seed_sequence or "",
            max_mutations=cfg.generation.max_mutations,
            feedback=feedback,
            existing_titles=existing_titles or [],
            count=count,
        )
        out: list[Hypothesis] = []
        for item in data.get("hypotheses", [])[:count]:
            seq = item.get("sequence")
            if seq:
                seq = "".join(c for c in seq.upper() if c.isalpha())
            h = Hypothesis(
                title=item.get("title", "Untitled"),
                summary=item.get("summary", ""),
                rationale=item.get("rationale", ""),
                sequence=seq or None,
                experiments=item.get("experiments", []) or [],
                round=round,
                origin="generation",
            )
            out.append(h)
            self.ctx.event(
                self.kind, "hypothesis", round=round, message=h.title,
                hypothesis_id=h.id, title=h.title,
            )
        return out
