"""Shared run-context passed to every agent.

Bundles the things agents commonly need: the LLM provider, the protein scorer,
the config, the store, the session, the domain string, and an event emitter so
each agent can stream progress to the CLI/web UI.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from coscientist.config import Config
from coscientist.core.models import AgentEvent, AgentKind, Session
from coscientist.core.store import Store
from coscientist.llm.base import LLMProvider


@dataclass
class RunContext:
    session: Session
    config: Config
    llm: LLMProvider
    store: Store
    scorer: object  # coscientist.protein.scorer.Scorer (avoid hard import here)
    domain: str = "computational biology and protein engineering"
    emit: Callable[[AgentEvent], None] = field(default=lambda ev: None)

    def event(self, agent: AgentKind, kind: str, round: int = 0, message: str = "", **data) -> None:
        ev = AgentEvent(
            session_id=self.session.id,
            agent=agent,
            round=round,
            kind=kind,
            message=message,
            data=data,
        )
        self.store.save_event(ev)
        self.emit(ev)
