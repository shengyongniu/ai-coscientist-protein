import math

from coscientist.core import tournament as T
from coscientist.core.models import Hypothesis


def test_expected_score_symmetry():
    assert math.isclose(T.expected_score(1200, 1200), 0.5)
    assert T.expected_score(1400, 1200) > 0.5
    assert T.expected_score(1000, 1200) < 0.5


def test_update_elo_conserves_points():
    a, b = 1200.0, 1200.0
    na, nb = T.update_elo(a, b, a_won=True)
    # Elo is zero-sum: total rating is conserved.
    assert math.isclose(na + nb, a + b, abs_tol=1e-6)
    assert na > a and nb < b


def test_apply_result_updates_winloss():
    a = Hypothesis(title="A", summary="")
    b = Hypothesis(title="B", summary="")
    T.apply_result(a, b, a.id)
    assert a.wins == 1 and a.losses == 0
    assert b.losses == 1 and b.wins == 0
    assert a.elo > b.elo


def test_make_pairings_no_self_pairs_and_capped():
    hs = [Hypothesis(title=f"H{i}", summary="") for i in range(6)]
    pairs = T.make_pairings(hs, max_matches=4)
    assert len(pairs) <= 4
    for x, y in pairs:
        assert x != y
    # no duplicate unordered pairs
    seen = {frozenset(p) for p in pairs}
    assert len(seen) == len(pairs)


def test_leaderboard_sorted_and_active_only():
    hs = [Hypothesis(title="A", summary="", elo=1300),
          Hypothesis(title="B", summary="", elo=1500),
          Hypothesis(title="C", summary="", elo=1100, active=False)]
    board = T.leaderboard(hs)
    assert [h.title for h in board] == ["B", "A"]
