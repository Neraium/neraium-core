from collections import deque


class SignalBuffer:
    def __init__(self, maxlen=60):
        self.buffer = deque(maxlen=maxlen)

    def add(self, signals: dict):
        self.buffer.append(signals)

    def latest(self):
        if not self.buffer:
            return None
        return self.buffer[-1]

    def window(self):
        return list(self.buffer)
