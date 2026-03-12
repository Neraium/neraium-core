from neraium_core.alignment import AlignmentEngine
from neraium_core.baseline import BASELINE_SYSTEM
from neraium_core.buffer import SignalBuffer
from neraium_core.scoring import ScoringEngine
from neraium_core.telemetry import TelemetryPayload


class TelemetryPipeline:
    def __init__(self):
        self.engine = AlignmentEngine(BASELINE_SYSTEM)
        self.buffer = SignalBuffer()
        self.scorer = ScoringEngine()

    def process(self, payload: TelemetryPayload):
        self.buffer.add(payload.signals)
        aligned = self.engine.align(payload.signals)
        score_result = self.scorer.score(aligned)

        return {
            "aligned": aligned,
            "score": score_result.score,
            "status": score_result.status,

        }
