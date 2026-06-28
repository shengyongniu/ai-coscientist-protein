import tempfile

import pytest

from coscientist.engine import build_context, run_session_async


@pytest.mark.asyncio
async def test_end_to_end_mocked_run():
    ctx, store = build_context(
        goal="Design improved nanobody binders for a target antigen",
        config="protein_binder",
        provider="mock",
        scorer="heuristic",
        rounds=2,
        db_path=tempfile.mktemp(suffix=".db"),
    )
    overview = await run_session_async(ctx)

    # Overview is well-formed.
    assert overview.summary
    assert overview.markdown.startswith("# Research Overview")
    assert len(overview.top_hypotheses) > 0

    # Hypotheses were generated, scored, ranked, and persisted.
    hs = store.get_hypotheses(ctx.session.id)
    assert len(hs) >= ctx.config.generation.initial_count
    scored = [h for h in hs if h.protein_score is not None]
    assert len(scored) > 0  # protein_mode -> sequences scored
    assert any(h.matches > 0 for h in hs)  # tournament ran

    # Evolution produced children in later rounds.
    assert any(h.origin.startswith("evolution") for h in hs)

    # Reviews + events were stored.
    assert len(store.get_reviews(ctx.session.id)) > 0
    assert len(store.get_events(ctx.session.id)) > 0

    # Session marked done.
    assert store.get_session(ctx.session.id).state == "done"


@pytest.mark.asyncio
async def test_non_protein_run_has_no_sequences():
    ctx, store = build_context(
        goal="Propose mechanisms for drug repurposing in ALS",
        config="default",
        provider="mock",
        scorer="heuristic",
        rounds=1,
        db_path=tempfile.mktemp(suffix=".db"),
    )
    overview = await run_session_async(ctx)
    assert overview.top_hypotheses
    # default config is not protein_mode; generation returns null sequences.
    hs = store.get_hypotheses(ctx.session.id)
    assert all(h.protein_score is None for h in hs)
