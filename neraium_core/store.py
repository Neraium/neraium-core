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
        return [
            event for event in self.events
            if event.get("anomaly", {}).get("anomaly") is True
        ]

    def structural_summary(self):
        if not self.events:
            return {
                "total_events": 0,
                "total_structural_anomalies": 0,
                "latest_status": "no data",
                "latest_drift_score": 0,
                "latest_vector": [],
                "relationship_points": [],
            }

        latest = self.events[-1]
        relationship_points = []

        for event in self.events[-50:]:
            signals = event.get("signals", {})
            relationship_points.append({
                "cpu_usage": signals.get("cpu_usage", 0),
                "memory_usage": signals.get("memory_usage", 0),
                "score": event.get("score", 0),
                "status": event.get("status", "normal"),
            })

        return {
            "total_events": len(self.events),
            "total_structural_anomalies": len(self.anomalies()),
            "latest_status": latest.get("status", "normal"),
            "latest_drift_score": latest.get("score", 0),
            "latest_vector": latest.get("aligned", []),
            "relationship_points": relationship_points,
        }
