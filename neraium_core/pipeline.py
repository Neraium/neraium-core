from datetime import datetime, timezone
import numpy as np

from neraium_core.features import MicroFeatureEngine


class TelemetryPipeline:

    def __init__(self):

        self.features = MicroFeatureEngine()

        self.history = []
        self.baseline_mean = None
        self.baseline_cov = None

        self.training_samples = 50

    def _update_baseline(self, vector):

        self.history.append(vector)

        if len(self.history) < self.training_samples:
            return

        data = np.array(self.history[-self.training_samples:])

        self.baseline_mean = np.mean(data, axis=0)
        self.baseline_cov = np.cov(data, rowvar=False)

    def _mahalanobis(self, vector):

        if self.baseline_mean is None:
            return 0

        x = np.array(vector)

        diff = x - self.baseline_mean

        try:
            inv_cov = np.linalg.pinv(self.baseline_cov)
        except:
            return 0

        score = np.sqrt(diff.T @ inv_cov @ diff)

        return float(score)

    def process(self, payload):

        cpu = payload.signals["cpu_usage"]
        mem = payload.signals["memory_usage"]

        f = self.features.compute(cpu, mem)

        vector = [
            f["cpu"],
            f["memory"],
            f["cpu_delta"],
            f["mem_delta"],
            f["cpu_std"],
            f["mem_std"]
        ]

        self._update_baseline(vector)

        score = self._mahalanobis(vector)

        if score > 4:
            status = "anomaly"
        else:
            status = "normal"

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signals": {
                "cpu_usage": cpu,
                "memory_usage": mem
            },
            "score": score,

            "status": status
        }

        return event
