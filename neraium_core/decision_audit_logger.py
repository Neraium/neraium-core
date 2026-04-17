"""Lightweight decision-audit logger for replay validation.

Writes one CSV row per frame containing decision layer metrics for inspection.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _sanitize_for_csv(text: str) -> str:
    """Remove emoji and control characters for Windows-safe CSV output."""
    if not isinstance(text, str):
        return str(text)
    # Remove emoji (U+1F300-U+1F9FF range) and control characters
    text = re.sub(r'[\U0001F300-\U0001F9FF\x00-\x1f\x7f-\x9f]', '', text)
    return text.strip()


class DecisionAuditLogger:
    """Writes decision audit rows to CSV during replay."""

    AUDIT_FIELDS = [
        "unit_id",
        "timestamp",
        "cycle",
        "state",
        "phase",
        "structural_drift_score",
        "composite_instability",
        "decision.finding_confidence",
        "decision.action_confidence",
        "decision.transient_score",
        "decision.suppress",
        "decision.severity",
        "decision.recommended_action",
        "decision.recommended_target",
        "decision.summary",
        "decision.reasons",
    ]

    def __init__(self, output_path: Path | str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.output_path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.AUDIT_FIELDS)
        self.writer.writeheader()
        self.row_count = 0
        logger.info(f"Decision audit logger initialized: {self.output_path}")

    def write_frame(self, frame_data: dict[str, Any]) -> None:
        """Write one audit row from frame data."""
        row = {}
        for field in self.AUDIT_FIELDS:
            row[field] = self._extract_field(frame_data, field)

        self.writer.writerow(row)
        self.row_count += 1

    def close(self) -> None:
        """Flush and close the file."""
        self.file.flush()
        self.file.close()
        logger.info(f"Decision audit log closed: {self.row_count} rows written")

    @staticmethod
    def _extract_field(frame_data: dict[str, Any], field: str) -> str:
        """Extract a field value from frame data, handling nested decision fields."""
        if "." not in field:
            value = frame_data.get(field)
            return "" if value is None else str(value)

        parts = field.split(".", 1)
        if parts[0] == "decision":
            decision = frame_data.get("decision", {})
            if isinstance(decision, dict):
                value = decision.get(parts[1])
                if value is None:
                    return ""
                if isinstance(value, list):
                    sanitized = [_sanitize_for_csv(str(v)) if parts[1] in ("summary", "reasons") else str(v) for v in value]
                    return ";".join(sanitized)
                result = str(value)
                if parts[1] in ("summary", "reasons"):
                    result = _sanitize_for_csv(result)
                return result
            return ""

        return ""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
