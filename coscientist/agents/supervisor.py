"""Supervisor planning helper: parses the goal into a research plan.

The heavy orchestration (scheduling, round loop) lives in
``coscientist.core.supervisor``; this class only owns the LLM planning call.
"""

from __future__ import annotations

from coscientist.agents.base import Agent
from coscientist.core.models import AgentKind


class SupervisorAgent(Agent):
    kind = AgentKind.SUPERVISOR

    def plan(self) -> dict:
        cfg = self.ctx.config
        try:
            data = self._call_json(
                "supervisor.system.j2",
                "supervisor.user.j2",
                model="strong",
                temperature=0.3,
                max_tokens=800,
                goal=cfg.goal or self.ctx.session.goal,
                protein_mode=cfg.protein_mode,
            )
        except ValueError:
            data = {"interpretation": cfg.goal or self.ctx.session.goal, "subgoals": []}
        self.ctx.event(
            self.kind, "plan", message="Research plan ready",
            interpretation=data.get("interpretation", ""),
            subgoals=data.get("subgoals", []),
        )
        return data
