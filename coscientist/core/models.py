"""Core data schemas for the AI Co-Scientist.

These pydantic models are the shared vocabulary across agents, the store, the
tournament, the CLI, and the web UI. Keeping them centralized makes the data
flow easy to follow and serialize.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> float:
    return time.time()


class AgentKind(str, Enum):
    SUPERVISOR = "supervisor"
    GENERATION = "generation"
    REFLECTION = "reflection"
    PROXIMITY = "proximity"
    RANKING = "ranking"
    EVOLUTION = "evolution"
    META_REVIEW = "meta_review"


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Hypothesis(BaseModel):
    """A candidate research hypothesis.

    For protein design the `sequence` field holds the amino-acid sequence under
    consideration; for general research goals it may be empty and the idea lives
    purely in `title`/`summary`.
    """

    id: str = Field(default_factory=lambda: _new_id("hyp"))
    title: str
    summary: str
    # Detailed rationale / mechanism the agent proposes.
    rationale: str = ""
    # Optional protein sequence this hypothesis proposes (amino acids).
    sequence: str | None = None
    # Free-form proposed experiments to validate the hypothesis.
    experiments: list[str] = Field(default_factory=list)

    # Provenance.
    round: int = 0
    parent_ids: list[str] = Field(default_factory=list)
    origin: str = "generation"  # generation | evolution:<strategy>

    # Elo rating maintained by the tournament.
    elo: float = 1200.0
    wins: int = 0
    losses: int = 0

    # Objective protein score (higher is better) and which scorer produced it.
    protein_score: float | None = None
    scorer_name: str | None = None
    score_detail: dict[str, Any] = Field(default_factory=dict)

    # Aggregated review outcome (filled by reflection).
    review_id: str | None = None
    active: bool = True  # set False when deduped/pruned by proximity

    created_at: float = Field(default_factory=_now)

    @property
    def matches(self) -> int:
        return self.wins + self.losses


class Review(BaseModel):
    """A structured peer review produced by the Reflection agent."""

    id: str = Field(default_factory=lambda: _new_id("rev"))
    hypothesis_id: str
    # Scores in [0, 10].
    correctness: float
    novelty: float
    testability: float
    safety: float
    critique: str
    suggestions: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=_now)

    @property
    def aggregate(self) -> float:
        return round((self.correctness + self.novelty + self.testability + self.safety) / 4.0, 3)


class DebateResult(BaseModel):
    """Outcome of a single pairwise tournament match (simulated debate)."""

    id: str = Field(default_factory=lambda: _new_id("match"))
    round: int
    hypothesis_a: str
    hypothesis_b: str
    winner: str  # one of the two ids
    reasoning: str = ""
    created_at: float = Field(default_factory=_now)


class Task(BaseModel):
    """A unit of work scheduled by the Supervisor onto the async worker queue."""

    id: str = Field(default_factory=lambda: _new_id("task"))
    kind: AgentKind
    round: int = 0
    state: TaskState = TaskState.PENDING
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: float = Field(default_factory=_now)


class AgentEvent(BaseModel):
    """A streamable event emitted as agents work, consumed by CLI + web UI."""

    id: str = Field(default_factory=lambda: _new_id("evt"))
    session_id: str
    agent: AgentKind
    round: int
    kind: str  # e.g. "start", "log", "hypothesis", "review", "match", "leaderboard", "done"
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=_now)


class ResearchOverview(BaseModel):
    """The final synthesized output produced by the Meta-review agent."""

    session_id: str
    goal: str
    summary: str
    top_hypotheses: list[Hypothesis] = Field(default_factory=list)
    recommended_experiments: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    markdown: str = ""
    created_at: float = Field(default_factory=_now)


class Session(BaseModel):
    """A full co-scientist run for a single research goal."""

    id: str = Field(default_factory=lambda: _new_id("sess"))
    goal: str
    config_name: str = "default"
    rounds: int = 3
    state: str = "created"  # created | running | done | failed
    created_at: float = Field(default_factory=_now)
