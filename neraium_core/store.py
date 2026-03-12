class EventStore:
    def __init__(self):
        self.events = []

    def add(self, event: dict):
        self.events.append(event)

    def all(self):
        return list(self.events)

    def latest(self):
        if not self.events:
            return None
        return self.events[-1]

    def anomalies(self):
        return [event for event in self.events if event.get("anomaly", {}).get("anomaly") is True]
