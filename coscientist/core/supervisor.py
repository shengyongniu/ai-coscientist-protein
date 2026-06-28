"""Supervisor: the async orchestrator running the long-horizon round loop.

Implements the generate -> reflect -> score -> dedup -> rank -> evolve cycle for
N rounds, then meta-review. Agent steps that are independent (reviewing each
hypothesis, scoring each sequence) run concurrently on a bounded worker pool via
``asyncio.to_thread`` (the LLM/torch calls are blocking I/O / CPU). Everything is
persisted to the store as it happens, so runs are resumable and streamable.
"""

from __future__ import annotations

import asyncio

from coscientist.agents.evolution import EvolutionAgent
from coscientist.agents.generation import GenerationAgent
from coscientist.agents.meta_review import MetaReviewAgent
from coscientist.agents.proximity import ProximityAgent
from coscientist.agents.ranking import RankingAgent
from coscientist.agents.reflection import ReflectionAgent
from coscientist.agents.supervisor import SupervisorAgent
from coscientist.core import tournament
from coscientist.core.context import RunContext
from coscientist.core.models import AgentKind, Hypothesis, Review


class Supervisor:
    def __init__(self, ctx: RunContext, concurrency: int = 4):
        self.ctx = ctx
        self.sem = asyncio.Semaphore(concurrency)
        self.gen = GenerationAgent(ctx)
        self.ref = ReflectionAgent(ctx)
        self.prox = ProximityAgent(ctx)
        self.rank = RankingAgent(ctx)
        self.evo = EvolutionAgent(ctx)
        self.meta = MetaReviewAgent(ctx)
        self.planner = SupervisorAgent(ctx)
        self.hypotheses: list[Hypothesis] = []
        self.reviews: list[Review] = []

    async def _bounded(self, fn, *args, **kwargs):
        async with self.sem:
            return await asyncio.to_thread(fn, *args, **kwargs)

    async def _score(self, h: Hypothesis) -> None:
        if not h.sequence:
            return
        res = await self._bounded(self.ctx.scorer.score, h.sequence)
        h.protein_score = res.score
        h.scorer_name = self.ctx.scorer.name
        h.score_detail = res.detail

    async def _review(self, h: Hypothesis, round: int) -> Review:
        review = await self._bounded(self.ref.review, h, round)
        return review

    def _persist(self) -> None:
        self.ctx.store.save_hypotheses(self.ctx.session.id, self.hypotheses)
        for r in self.reviews:
            self.ctx.store.save_review(self.ctx.session.id, r)

    async def run(self):
        ctx = self.ctx
        cfg = ctx.config
        session = ctx.session
        session.state = "running"
        ctx.store.save_session(session)

        # Plan (Supervisor agent).
        await self._bounded(self.planner.plan)

        feedback = ""
        for rnd in range(cfg.rounds):
            ctx.event(AgentKind.SUPERVISOR, "round_start", round=rnd, message=f"Round {rnd + 1}/{cfg.rounds}")

            # --- Generation / Evolution ---
            if rnd == 0:
                new = await self._bounded(self.gen.generate, cfg.generation.initial_count, rnd, feedback)
            else:
                parents = tournament.leaderboard(self.hypotheses, top_k=5)
                strategy = cfg.evolution.strategies[(rnd - 1) % len(cfg.evolution.strategies)]
                new = await self._bounded(
                    self.evo.evolve, parents, strategy, cfg.generation.evolve_count, rnd, feedback
                )
            self.hypotheses.extend(new)

            # --- Score new sequences concurrently ---
            await asyncio.gather(*(self._score(h) for h in new))

            # --- Reflect on new hypotheses concurrently ---
            new_reviews = await asyncio.gather(*(self._review(h, rnd) for h in new))
            self.reviews.extend(new_reviews)

            # --- Proximity dedup + clustering ---
            _, groups = await self._bounded(
                self.prox.deduplicate, self.hypotheses, cfg.proximity.dedup_threshold, rnd
            )

            # --- Ranking tournament ---
            await self._bounded(self.rank.run_tournament, self.hypotheses, rnd, groups)

            # --- Feedback for next round (top critiques) ---
            feedback = self._build_feedback()
            self._persist()
            ctx.event(AgentKind.SUPERVISOR, "round_end", round=rnd, message=f"Round {rnd + 1} complete")

        # --- Meta-review ---
        top = tournament.leaderboard(self.hypotheses, top_k=10)
        overview = await self._bounded(self.meta.synthesize, top, self.reviews, cfg.rounds)

        session.state = "done"
        ctx.store.save_session(session)
        return overview

    def _build_feedback(self) -> str:
        """Aggregate the lowest-scoring critique themes to guide the next round."""
        rev_by_id = {r.hypothesis_id: r for r in self.reviews}
        top = tournament.leaderboard(self.hypotheses, top_k=5)
        lines = []
        for h in top:
            r = rev_by_id.get(h.id)
            if r and r.critique:
                lines.append(f"- {h.title}: {r.critique}")
        return "Address these critiques of the current leaders:\n" + "\n".join(lines) if lines else ""
