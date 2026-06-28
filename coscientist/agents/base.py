"""Base agent: shared LLM-call + JSON-parsing helpers."""

from __future__ import annotations

from coscientist.core.context import RunContext
from coscientist.core.models import AgentKind
from coscientist.llm.base import extract_json
from coscientist.llm.prompts import render


class Agent:
    kind: AgentKind = AgentKind.SUPERVISOR

    def __init__(self, ctx: RunContext):
        self.ctx = ctx

    def _call_json(
        self,
        system_template: str,
        user_template: str,
        *,
        model: str = "strong",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        retries: int = 2,
        **context,
    ):
        """Render prompts, call the LLM, and parse JSON with a small retry loop."""
        system = render(system_template, **context)
        user = render(user_template, **context)
        last_err: Exception | None = None
        for _ in range(retries + 1):
            text = self.ctx.llm.complete_text(
                system, user, model=model, temperature=temperature, max_tokens=max_tokens
            )
            try:
                return extract_json(text)
            except ValueError as e:
                last_err = e
                continue
        raise ValueError(f"{self.kind.value} agent could not parse JSON: {last_err}")
