from neraium_core.alignment import AlignmentEngine

def test_alignment_basic():
    engine = AlignmentEngine()
    score = engine.score(0.9, 0.9)

    assert score >= 0
