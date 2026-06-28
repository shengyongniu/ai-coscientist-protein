from coscientist.llm.base import extract_json
from coscientist.llm.mock import MockProvider
from coscientist.protein.scorer import HeuristicScorer


def test_extract_json_fenced():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_with_prose():
    assert extract_json('Here you go: {"x": [1,2]} thanks') == {"x": [1, 2]}


def test_extract_json_raises_on_garbage():
    import pytest

    with pytest.raises(ValueError):
        extract_json("no json here at all")


def test_mock_generation_shape():
    p = MockProvider()
    out = p.complete_text("[[MOCK:generation]]", "Generate exactly 4 variants. SEED_SEQUENCE: ACDEFGHIKLMNPQRSTVWY")
    data = extract_json(out)
    assert len(data["hypotheses"]) == 4
    assert all("title" in h for h in data["hypotheses"])


def test_mock_is_deterministic():
    p1, p2 = MockProvider(), MockProvider()
    a = p1.complete_text("[[MOCK:ranking]]", "A vs B same prompt")
    b = p2.complete_text("[[MOCK:ranking]]", "A vs B same prompt")
    assert a == b


def test_usage_tracking():
    p = MockProvider()
    p.complete_text("[[MOCK:reflection]]", "review x")
    s = p.usage.summary()
    assert s["calls"] == 1 and s["output_tokens"] > 0


def test_heuristic_scorer_range_and_validity():
    s = HeuristicScorer()
    good = s.score("QVQLVESGGGLVQAGGSLRLSCAASGRTFSEYAMGW")
    empty = s.score("")
    assert 0.0 <= good.score <= 1.0
    assert empty.score == 0.0
    assert good.detail["length"] == 36
