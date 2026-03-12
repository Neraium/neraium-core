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
                "baseline_centroid": {"x": 0, "y": 0},
                "drift_threshold": 30,
                "trajectory_vectors": [],
            }

        recent = self.events[-50:]
        anomalies = self.anomalies()
        latest = self.events[-1]

        relationship_points = []
        for event in recent:
            signals = event.get("signals", {})
            relationship_points.append({
                "x": signals.get("cpu_usage", 0),
                "y": signals.get("memory_usage", 0),
                "score": event.get("score", 0),
                "status": event.get("status", "normal"),
                "timestamp": event.get("timestamp", ""),
            })

        if relationship_points:
            centroid_x = sum(p["x"] for p in relationship_points) / len(relationship_points)
            centroid_y = sum(p["y"] for p in relationship_points) / len(relationship_points)
        else:
            centroid_x = 0
            centroid_y = 0

        trajectory_vectors = []
        for i in range(1, len(relationship_points)):
            prev_point = relationship_points[i - 1]
            curr_point = relationship_points[i]
            trajectory_vectors.append({
                "x1": prev_point["x"],
                "y1": prev_point["y"],
                "x2": curr_point["x"],
                "y2": curr_point["y"],
                "status": curr_point["status"],
            })

        return {
            "total_events": len(self.events),
            "total_structural_anomalies": len(anomalies),
            "latest_status": latest.get("status", "normal"),
            "latest_drift_score": latest.get("score", 0),
            "latest_vector": latest.get("aligned", []),
            "relationship_points": relationship_points,
            "baseline_centroid": {
                "x": centroid_x,
                "y": centroid_y,
            },
            "drift_threshold": 30,
            "trajectory_vectors": trajectory_vectors,
        }
