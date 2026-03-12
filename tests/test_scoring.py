from neraium_core.scoring import ScoringEngine


def test_score_normal():
    engine = ScoringEngine(threshold=80.0)
    result = engine.score([20.0, 35.0])

    assert result.status == "normal"
    assert result.score == 35.0


def test_score_anomaly():
    engine = ScoringEngine(threshold=80.0)
    result = engine.score([20.0, 95.0])

    assert result.status == "anomaly"
    assert result.score == 95.0


def test_score_empty():
    engine = ScoringEngine()
    result = engine.score([])

    assert result.status == "empty"
    assert result.score == 0.0
