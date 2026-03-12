import numpy as np


class ScoringEngine:
    """
    Structural scoring engine based on geometric drift
    between system state vectors.
    """

    def __init__(self, history_window=20, drift_threshold=30):
        self.history = []
        self.history_window = history_window
        self.drift_threshold = drift_threshold

    def score(self, aligned_vector, signals=None):
        """
        aligned_vector example:
        [cpu_usage, memory_usage]
        """

        x = np.array(aligned_vector)

        # store history
        self.history.append(x)

        # warm-up period
        if len(self.history) < self.history_window:
            return {
                "score": 0,
                "status": "normal",
                "anomaly": {
                    "anomaly": False,
                    "reason": "insufficient data"
                }
            }

        # compute baseline geometry
        baseline = np.mean(self.history[-self.history_window:], axis=0)

        # geometric drift
        drift = np.linalg.norm(x - baseline)

        anomaly = drift > self.drift_threshold

        anomaly_info = {
            "anomaly": anomaly
        }

        if anomaly:
            anomaly_info["drift"] = float(drift)

            if signals:
                cpu = signals.get("cpu_usage", 0)
                mem = signals.get("memory_usage", 0)

                cpu_base = baseline[0]
                mem_base = baseline[1]

                anomaly_info["cpu_delta"] = float(cpu - cpu_base)
                anomaly_info["mem_delta"] = float(mem - mem_base)

        return {
            "score": float(drift),
            "status": "anomaly" if anomaly else "normal",
            "anomaly": anomaly_info
        }
