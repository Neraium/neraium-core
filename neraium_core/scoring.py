from pydantic import BaseModel


class ScoreResult(BaseModel):
    score: float
    status: str


class ScoringEngine:
    def __init__(self, threshold: float = 80.0):
        self.threshold = threshold

    def score(self, vector):
        if not vector:
            return ScoreResult(score=0.0, status="empty")

        score = max(float(v) for v in vector)

        if score >= self.threshold:
            return ScoreResult(score=score, status="anomaly")

        return ScoreResult(score=score, status="normal")
