from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from neraium_core.alignment import AlignmentEngine
from neraium_core.scoring import mahalanobis_distance


class ScoredWindow(BaseModel):
    system_id: str
    window_end: datetime
    vector: list[float | None]
    accepted_for_scoring: bool
    score: Optional[float] = None
    reason: Optional[str] = None


class NeraiumPipeline:
    def __init__(self, system_definition, baseline):
        self.system_definition = system_definition
        self.baseline = baseline
        self.engine = AlignmentEngine(system_definition)

        if baseline.system_id != system_definition.system_id:
            raise ValueError("baseline/system mismatch: system_id does not match")

        if baseline.signal_order != list(system_definition.vector_order):
            raise ValueError("baseline/system mismatch: vector_order does not match signal_order")

    def ingest(self, payload):
        emitted = self.engine.ingest(payload)
        results = []

        for window in emitted:
            vector = list(window.vector)

            if not window.accepted_for_scoring:
                results.append(
                    ScoredWindow(
                        system_id=window.system_id,
                        window_end=window.window_end,
                        vector=vector,
                        accepted_for_scoring=False,
                        score=None,
                        reason="window rejected for scoring",
                    )
                )
                continue

            if any(v is None for v in vector):
                results.append(
                    ScoredWindow(
                        system_id=window.system_id,
                        window_end=window.window_end,
                        vector=vector,
                        accepted_for_scoring=False,
                        score=None,
                        reason="missing values remain after forward fill",
                    )
                )
                continue

            score = mahalanobis_distance(
                vector=[float(v) for v in vector],
                mean_vector=self.baseline.mean_vector,
                covariance_matrix=self.baseline.covariance_matrix,
            )

            results.append(
                ScoredWindow(
                    system_id=window.system_id,
                    window_end=window.window_end,
                    vector=[float(v) for v in vector],
                    accepted_for_scoring=True,
                    score=float(score),
                    reason=None,
                )
            )

        return results