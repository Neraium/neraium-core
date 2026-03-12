import sqlite3
import json


class EventStore:
    def __init__(self, db_path="neraium_events.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            status TEXT,
            score REAL,
            signals TEXT,
            features TEXT,
            aligned TEXT,
            anomaly TEXT
        )
        """)
        self.conn.commit()

    def add(self, event: dict):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO events (timestamp, status, score, signals, features, aligned, anomaly)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.get("timestamp"),
                event.get("status"),
                event.get("score"),
                json.dumps(event.get("signals", {})),
                json.dumps(event.get("features", {})),
                json.dumps(event.get("aligned", [])),
                json.dumps(event.get("anomaly", {})),
            ),
        )
        self.conn.commit()

    def all(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT timestamp, status, score, signals, features, aligned, anomaly
        FROM events
        ORDER BY id ASC
        """)
        rows = cursor.fetchall()

        events = []
        for row in rows:
            events.append({
                "timestamp": row[0],
                "status": row[1],
                "score": row[2],
                "signals": json.loads(row[3]),
                "features": json.loads(row[4]),
                "aligned": json.loads(row[5]),
                "anomaly": json.loads(row[6]),
            })
        return events

    def latest(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT timestamp, status, score, signals, features, aligned, anomaly
        FROM events
        ORDER BY id DESC
        LIMIT 1
        """)
        row = cursor.fetchone()

        if not row:
            return None

        return {
            "timestamp": row[0],
            "status": row[1],
            "score": row[2],
            "signals": json.loads(row[3]),
            "features": json.loads(row[4]),
            "aligned": json.loads(row[5]),
            "anomaly": json.loads(row[6]),
        }

    def anomalies(self):
        return [
            event for event in self.all()
            if event.get("status") == "anomaly"
        ]

    def structural_summary(self):
        events = self.all()

        if not events:
            return {
                "total_events": 0,
                "total_structural_anomalies": 0,
                "latest_status": "no data",
                "latest_drift_score": 0,
                "latest_vector": [],
                "relationship_points": [],
                "baseline_centroid": {"x": 0, "y": 0},
                "drift_threshold": 30,
            }

        anomalies = [e for e in events if e["status"] == "anomaly"]
        latest = events[-1]

        relationship_points = []
        for e in events[-50:]:
            relationship_points.append({
                "x": e["signals"].get("cpu_usage", 0),
                "y": e["signals"].get("memory_usage", 0),
                "status": e["status"],
                "timestamp": e["timestamp"],
                "score": e["score"],
            })

        cx = sum(p["x"] for p in relationship_points) / len(relationship_points)
        cy = sum(p["y"] for p in relationship_points) / len(relationship_points)

        return {
            "total_events": len(events),
            "total_structural_anomalies": len(anomalies),
            "latest_status": latest["status"],
            "latest_drift_score": latest["score"],
            "latest_vector": latest["aligned"],
            "relationship_points": relationship_points,
            "baseline_centroid": {"x": cx, "y": cy},
            "drift_threshold": 30,
        }
