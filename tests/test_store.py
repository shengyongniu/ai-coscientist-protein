import tempfile

from coscientist.core.models import (
    AgentEvent,
    AgentKind,
    DebateResult,
    Hypothesis,
    Review,
    Session,
)
from coscientist.core.store import Store


def _store():
    return Store(tempfile.mktemp(suffix=".db"))


def test_session_roundtrip():
    s = _store()
    sess = Session(goal="g", rounds=3)
    s.save_session(sess)
    got = s.get_session(sess.id)
    assert got is not None and got.goal == "g" and got.rounds == 3
    assert sess.id in [x.id for x in s.list_sessions()]


def test_hypothesis_persist_and_order_by_elo():
    s = _store()
    sess = Session(goal="g")
    s.save_session(sess)
    s.save_hypotheses(sess.id, [
        Hypothesis(title="low", summary="", elo=1100),
        Hypothesis(title="high", summary="", elo=1400),
    ])
    hs = s.get_hypotheses(sess.id)
    assert [h.title for h in hs] == ["high", "low"]


def test_active_only_filter():
    s = _store()
    sess = Session(goal="g")
    s.save_session(sess)
    s.save_hypotheses(sess.id, [
        Hypothesis(title="a", summary="", active=True),
        Hypothesis(title="b", summary="", active=False),
    ])
    assert len(s.get_hypotheses(sess.id, active_only=True)) == 1


def test_reviews_matches_events():
    s = _store()
    sess = Session(goal="g")
    s.save_session(sess)
    h = Hypothesis(title="a", summary="")
    s.save_hypothesis(sess.id, h)
    s.save_review(sess.id, Review(hypothesis_id=h.id, correctness=8, novelty=7, testability=6, safety=9, critique="c"))
    s.save_match(sess.id, DebateResult(round=0, hypothesis_a=h.id, hypothesis_b=h.id, winner=h.id))
    s.save_event(AgentEvent(session_id=sess.id, agent=AgentKind.GENERATION, round=0, kind="log", message="hi"))
    assert len(s.get_reviews(sess.id)) == 1
    assert len(s.get_matches(sess.id)) == 1
    evs = s.get_events(sess.id)
    assert len(evs) == 1 and evs[0]["message"] == "hi"


def test_events_after_seq():
    s = _store()
    sess = Session(goal="g")
    s.save_session(sess)
    for i in range(3):
        s.save_event(AgentEvent(session_id=sess.id, agent=AgentKind.RANKING, round=0, kind="log", message=str(i)))
    first = s.get_events(sess.id)
    assert len(first) == 3
    tail = s.get_events(sess.id, after_seq=first[0]["_seq"])
    assert len(tail) == 2


def test_review_aggregate():
    r = Review(hypothesis_id="x", correctness=8, novelty=6, testability=10, safety=8, critique="")
    assert r.aggregate == 8.0
