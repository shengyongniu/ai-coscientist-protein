"""Durable SQLite-backed store for co-scientist sessions.

Persisting every hypothesis, review, match, task, and event makes long-horizon
runs resumable and gives the web UI a queryable history. The store is
intentionally simple: one row per object, JSON-encoded payloads, indexed by
session.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from coscientist.core.models import (
    AgentEvent,
    DebateResult,
    Hypothesis,
    Review,
    Session,
    Task,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    config_name TEXT,
    rounds INTEGER,
    state TEXT,
    created_at REAL,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    round INTEGER,
    elo REAL,
    active INTEGER,
    created_at REAL,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    hypothesis_id TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS matches (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    round INTEGER,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    state TEXT,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    seq INTEGER,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hyp_session ON hypotheses(session_id);
CREATE INDEX IF NOT EXISTS idx_rev_session ON reviews(session_id);
CREATE INDEX IF NOT EXISTS idx_match_session ON matches(session_id);
CREATE INDEX IF NOT EXISTS idx_task_session ON tasks(session_id);
CREATE INDEX IF NOT EXISTS idx_evt_session ON events(session_id, seq);
"""


class Store:
    """Thread-safe SQLite store. One instance per process is fine."""

    def __init__(self, path: str | Path = "data/coscientist.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._event_seq = 0

    # ---- sessions ----
    def save_session(self, session: Session) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO sessions(id, goal, config_name, rounds, state, created_at, data)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    session.id,
                    session.goal,
                    session.config_name,
                    session.rounds,
                    session.state,
                    session.created_at,
                    session.model_dump_json(),
                ),
            )
            self._conn.commit()

    def get_session(self, session_id: str) -> Session | None:
        row = self._conn.execute(
            "SELECT data FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        return Session.model_validate_json(row["data"]) if row else None

    def list_sessions(self) -> list[Session]:
        rows = self._conn.execute(
            "SELECT data FROM sessions ORDER BY created_at DESC"
        ).fetchall()
        return [Session.model_validate_json(r["data"]) for r in rows]

    # ---- hypotheses ----
    def save_hypothesis(self, session_id: str, h: Hypothesis) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO hypotheses(id, session_id, round, elo, active, created_at, data)"
                " VALUES (?,?,?,?,?,?,?)",
                (h.id, session_id, h.round, h.elo, int(h.active), h.created_at, h.model_dump_json()),
            )
            self._conn.commit()

    def save_hypotheses(self, session_id: str, hs: list[Hypothesis]) -> None:
        for h in hs:
            self.save_hypothesis(session_id, h)

    def get_hypotheses(self, session_id: str, active_only: bool = False) -> list[Hypothesis]:
        q = "SELECT data FROM hypotheses WHERE session_id=?"
        if active_only:
            q += " AND active=1"
        q += " ORDER BY elo DESC"
        rows = self._conn.execute(q, (session_id,)).fetchall()
        return [Hypothesis.model_validate_json(r["data"]) for r in rows]

    # ---- reviews ----
    def save_review(self, session_id: str, r: Review) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO reviews(id, session_id, hypothesis_id, data) VALUES (?,?,?,?)",
                (r.id, session_id, r.hypothesis_id, r.model_dump_json()),
            )
            self._conn.commit()

    def get_reviews(self, session_id: str) -> list[Review]:
        rows = self._conn.execute(
            "SELECT data FROM reviews WHERE session_id=?", (session_id,)
        ).fetchall()
        return [Review.model_validate_json(r["data"]) for r in rows]

    # ---- matches ----
    def save_match(self, session_id: str, m: DebateResult) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO matches(id, session_id, round, data) VALUES (?,?,?,?)",
                (m.id, session_id, m.round, m.model_dump_json()),
            )
            self._conn.commit()

    def get_matches(self, session_id: str) -> list[DebateResult]:
        rows = self._conn.execute(
            "SELECT data FROM matches WHERE session_id=?", (session_id,)
        ).fetchall()
        return [DebateResult.model_validate_json(r["data"]) for r in rows]

    # ---- tasks ----
    def save_task(self, session_id: str, t: Task) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO tasks(id, session_id, state, data) VALUES (?,?,?,?)",
                (t.id, session_id, t.state.value, t.model_dump_json()),
            )
            self._conn.commit()

    def get_tasks(self, session_id: str) -> list[Task]:
        rows = self._conn.execute(
            "SELECT data FROM tasks WHERE session_id=?", (session_id,)
        ).fetchall()
        return [Task.model_validate_json(r["data"]) for r in rows]

    # ---- events ----
    def save_event(self, ev: AgentEvent) -> None:
        with self._lock:
            self._event_seq += 1
            self._conn.execute(
                "INSERT OR REPLACE INTO events(id, session_id, seq, data) VALUES (?,?,?,?)",
                (ev.id, ev.session_id, self._event_seq, ev.model_dump_json()),
            )
            self._conn.commit()

    def get_events(self, session_id: str, after_seq: int = 0) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT seq, data FROM events WHERE session_id=? AND seq>? ORDER BY seq",
            (session_id, after_seq),
        ).fetchall()
        out = []
        for r in rows:
            d = json.loads(r["data"])
            d["_seq"] = r["seq"]
            out.append(d)
        return out

    def close(self) -> None:
        self._conn.close()
