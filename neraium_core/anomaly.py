from statistics import mean


class RollingAnomalyDetector:
    def __init__(self, window_size=5, threshold=30.0):
        self.window_size = window_size
        self.threshold = threshold

    def detect(self, window):
        """
        window: list of signal dicts from the buffer
        """
        if len(window) < self.window_size:
            return {"anomaly": False, "reason": "insufficient data"}

        cpu_values = [w["cpu_usage"] for w in window]
        mem_values = [w["memory_usage"] for w in window]

        cpu_avg = mean(cpu_values[:-1])
        mem_avg = mean(mem_values[:-1])

        latest_cpu = cpu_values[-1]
        latest_mem = mem_values[-1]

        cpu_delta = abs(latest_cpu - cpu_avg)
        mem_delta = abs(latest_mem - mem_avg)

        if cpu_delta > self.threshold or mem_delta > self.threshold:
            return {
                "anomaly": True,
                "cpu_delta": cpu_delta,
                "mem_delta": mem_delta,
            }

        return {"anomaly": False}
