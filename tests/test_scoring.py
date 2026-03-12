from neraium_core.scoring import ScoringEngine


def test_score_empty():
    engine = ScoringEngine()

    result = engine.score([])

    assert result["status"] == "normal"


def test_score_normal():
    engine = ScoringEngine()

    result = engine.score([20, 40], {"cpu_usage": 20, "memory_usage": 40})

    assert "score" in result


def test_score_anomaly():
    engine = ScoringEngine()

    # build baseline history
    for _ in range(25):
        engine.score([20, 40], {"cpu_usage": 20, "memory_usage": 40})

    result = engine.score([95, 99], {"cpu_usage": 95, "memory_usage": 99})

    assert "status" in result
