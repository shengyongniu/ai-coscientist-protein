"""Evolution agent: improves top hypotheses via rotating strategies."""

from __future__ import annotations

from coscientist.agents.base import Agent
from coscientist.core.models import AgentKind, Hypothesis


class EvolutionAgent(Agent):
    kind = AgentKind.EVOLUTION

    def evolve(
        self,
        parents: list[Hypothesis],
        strategy: str,
        count: int,
        round: int,
        feedback: str = "",
    ) -> list[Hypothesis]:
        cfg = self.ctx.config
        self.ctx.event(
            self.kind, "start", round=round,
            message=f"Evolving {len(parents)} parents via '{strategy}'",
        )
        data = self._call_json(
            "evolution.system.j2",
            "evolution.user.j2",
            model="strong",
            temperature=cfg.evolution.temperature,
            domain=self.ctx.domain,
            goal=cfg.goal or self.ctx.session.goal,
            protein_mode=cfg.protein_mode,
            seed_sequence=cfg.seed_sequence or "",
            parents=parents,
            strategy=strategy,
            feedback=feedback,
            count=count,
        )
        parent_ids = [p.id for p in parents]
        out: list[Hypothesis] = []
        for item in data.get("hypotheses", [])[:count]:
            seq = item.get("sequence")
            if seq:
                seq = "".join(c for c in seq.upper() if c.isalpha())
            h = Hypothesis(
                title=item.get("title", "Evolved hypothesis"),
                summary=item.get("summary", ""),
                rationale=item.get("rationale", ""),
                sequence=seq or None,
                experiments=item.get("experiments", []) or [],
                round=round,
                origin=f"evolution:{strategy}",
                parent_ids=parent_ids,
                # Seed Elo near the parents' average so it must prove itself.
                elo=sum(p.elo for p in parents) / len(parents) if parents else 1200.0,
            )
            out.append(h)
            self.ctx.event(
                self.kind, "hypothesis", round=round, message=h.title,
                hypothesis_id=h.id, title=h.title, strategy=strategy,
            )
        return out
