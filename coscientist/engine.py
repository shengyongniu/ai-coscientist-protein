"""Top-level engine: assemble a RunContext and run a full session."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from coscientist.config import Config, load_config
from coscientist.core.context import RunContext
from coscientist.core.models import AgentEvent, ResearchOverview, Session
from coscientist.core.store import Store
from coscientist.core.supervisor import Supervisor
from coscientist.llm import get_provider
from coscientist.protein import get_scorer

DEFAULT_DOMAIN = "computational biology and protein engineering"


def build_context(
    goal: str,
    config: str = "default",
    *,
    provider: str | None = None,
    scorer: str | None = None,
    rounds: int | None = None,
    db_path: str = "data/coscientist.db",
    emit: Callable[[AgentEvent], None] | None = None,
    domain: str = DEFAULT_DOMAIN,
) -> tuple[RunContext, Store]:
    cfg: Config = load_config(config)
    if rounds is not None:
        cfg.rounds = rounds
    # Goal precedence: explicit arg > config default.
    if goal:
        cfg.goal = goal
    goal = cfg.goal or goal

    store = Store(db_path)
    session = Session(goal=goal, config_name=cfg.name, rounds=cfg.rounds)
    store.save_session(session)

    ctx = RunContext(
        session=session,
        config=cfg,
        llm=get_provider(provider),
        store=store,
        scorer=get_scorer(scorer or cfg.scorer),
        domain=domain,
        emit=emit or (lambda ev: None),
    )
    return ctx, store


async def run_session_async(ctx: RunContext, concurrency: int = 4) -> ResearchOverview:
    supervisor = Supervisor(ctx, concurrency=concurrency)
    return await supervisor.run()


def run_session(
    goal: str,
    config: str = "default",
    *,
    provider: str | None = None,
    scorer: str | None = None,
    rounds: int | None = None,
    db_path: str = "data/coscientist.db",
    emit: Callable[[AgentEvent], None] | None = None,
    artifacts_dir: str = "data/artifacts",
) -> tuple[ResearchOverview, RunContext]:
    ctx, store = build_context(
        goal, config, provider=provider, scorer=scorer, rounds=rounds, db_path=db_path, emit=emit
    )
    overview = asyncio.run(run_session_async(ctx))
    _write_artifacts(ctx, overview, artifacts_dir)
    return overview, ctx


def _write_artifacts(ctx: RunContext, overview: ResearchOverview, artifacts_dir: str) -> None:
    out = Path(artifacts_dir) / ctx.session.id / "final"
    out.mkdir(parents=True, exist_ok=True)
    (out / "overview.md").write_text(overview.markdown)
    (out / "overview.json").write_text(overview.model_dump_json(indent=2))
