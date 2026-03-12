from collections import deque
import statistics


class MicroFeatureEngine:

    def __init__(self, window=10):
        self.window = window
        self.cpu_history = deque(maxlen=window)
        self.mem_history = deque(maxlen=window)

    def compute(self, cpu, mem):

        self.cpu_history.append(cpu)
        self.mem_history.append(mem)

        features = {
            "cpu": cpu,
            "memory": mem,
        }

        if len(self.cpu_history) > 1:
            features["cpu_delta"] = cpu - self.cpu_history[-2]
            features["mem_delta"] = mem - self.mem_history[-2]
        else:
            features["cpu_delta"] = 0
            features["mem_delta"] = 0

        if len(self.cpu_history) >= 3:
            features["cpu_std"] = statistics.stdev(self.cpu_history)
            features["mem_std"] = statistics.stdev(self.mem_history)
        else:
            features["cpu_std"] = 0
            features["mem_std"] = 0

        return features
