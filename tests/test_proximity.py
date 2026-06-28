from coscientist.agents.proximity import cosine, embed
from coscientist.core.models import Hypothesis


def test_identical_text_high_similarity():
    h1 = Hypothesis(title="Stability via core packing", summary="improve hydrophobic core", rationale="")
    h2 = Hypothesis(title="Stability via core packing", summary="improve hydrophobic core", rationale="")
    assert cosine(embed(h1), embed(h2)) > 0.99


def test_different_text_low_similarity():
    h1 = Hypothesis(title="electrostatic interface", summary="salt bridges at paratope", rationale="")
    h2 = Hypothesis(title="loop shortening", summary="entropy optimized truncation", rationale="")
    assert cosine(embed(h1), embed(h2)) < 0.5


def test_sequence_embedding_normalized():
    h = Hypothesis(title="x", summary="", sequence="ACDEFGHIKLMNPQRSTVWYACDEFGHIK")
    v = embed(h)
    import numpy as np

    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_similar_sequences_more_similar_than_different():
    base = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    near = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWA"  # 1 mutation
    far = "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"
    hb = Hypothesis(title="b", summary="", sequence=base)
    hn = Hypothesis(title="n", summary="", sequence=near)
    hf = Hypothesis(title="f", summary="", sequence=far)
    assert cosine(embed(hb), embed(hn)) > cosine(embed(hb), embed(hf))
