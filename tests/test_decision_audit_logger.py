"""Tests for decision audit logger."""

import csv
import tempfile
from pathlib import Path

import pytest

from neraium_core.decision_audit_logger import DecisionAuditLogger, _sanitize_for_csv


class TestSanitizeForCSV:
    """Test emoji and control character sanitization."""

    def test_sanitize_emoji(self):
        """Test that emoji are removed."""
        text = "Alert 🚨 detected"
        result = _sanitize_for_csv(text)
        assert "🚨" not in result
        assert result == "Alert detected"

    def test_sanitize_control_characters(self):
        """Test that control characters are removed."""
        text = "Normal\x00text\x1fwith\x7fcontrol"
        result = _sanitize_for_csv(text)
        assert "\x00" not in result
        assert "\x1f" not in result
        assert "\x7f" not in result
        assert result == "Normaltextwithcontrol"

    def test_sanitize_multiple_emojis(self):
        """Test multiple emojis are removed."""
        text = "🚨 Alert 🔴 Critical 🌡️"
        result = _sanitize_for_csv(text)
        assert "🚨" not in result
        assert "🔴" not in result
        assert "🌡️" not in result
        assert result == "Alert Critical"

    def test_sanitize_preserves_normal_text(self):
        """Test that normal text is preserved."""
        text = "Normal text without special characters"
        result = _sanitize_for_csv(text)
        assert result == text

    def test_sanitize_non_string_input(self):
        """Test that non-string input is converted to string."""
        result = _sanitize_for_csv(123)
        assert result == "123"

    def test_sanitize_strips_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        text = "  emoji test 🚨  "
        result = _sanitize_for_csv(text)
        assert result == "emoji test"
        assert not result.startswith(" ")
        assert not result.endswith(" ")


class TestDecisionAuditLogger:
    """Test decision audit logger functionality."""

    def test_logger_initialization(self):
        """Test that logger creates output file and writes header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "audit.csv"
            logger = DecisionAuditLogger(output_path)
            logger.close()

            assert output_path.exists()
            with open(output_path, "r", encoding="utf-8") as f:
                header = f.readline().strip()
                assert "unit_id" in header
                assert "decision.finding_confidence" in header

    def test_write_frame_basic(self):
        """Test writing a basic frame to the audit log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "audit.csv"
            logger = DecisionAuditLogger(output_path)

            frame_data = {
                "unit_id": "unit_001",
                "timestamp": 1234567890.0,
                "cycle": 1,
                "state": "ALERT",
                "phase": "degradation",
                "structural_drift_score": 0.85,
                "composite_instability": 0.75,
                "decision": {
                    "finding_confidence": 0.95,
                    "action_confidence": 0.90,
                    "transient_score": 0.1,
                    "suppress": False,
                    "severity": "high",
                    "recommended_action": "inspect",
                    "recommended_target": "bearing",
                    "summary": "High drift detected",
                    "reasons": ["sensor anomaly", "pattern mismatch"],
                },
            }

            logger.write_frame(frame_data)
            logger.close()

            # Verify the CSV was written correctly
            with open(output_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 1
                row = rows[0]
                assert row["unit_id"] == "unit_001"
                assert row["state"] == "ALERT"
                assert row["decision.severity"] == "high"

    def test_write_frame_with_emoji_in_decision(self):
        """Test that emoji in decision fields are sanitized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "audit.csv"
            logger = DecisionAuditLogger(output_path)

            frame_data = {
                "unit_id": "unit_001",
                "timestamp": 1234567890.0,
                "cycle": 1,
                "state": "ALERT",
                "phase": "degradation",
                "structural_drift_score": 0.85,
                "composite_instability": 0.75,
                "decision": {
                    "finding_confidence": 0.95,
                    "action_confidence": 0.90,
                    "transient_score": 0.1,
                    "suppress": False,
                    "severity": "high",
                    "recommended_action": "inspect",
                    "recommended_target": "bearing",
                    "summary": "Alert 🚨 detected with high confidence",
                    "reasons": ["anomaly 🔴 detected", "pattern 📊 mismatch"],
                },
            }

            logger.write_frame(frame_data)
            logger.close()

            # Verify emoji are removed from the CSV
            with open(output_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 1
                row = rows[0]
                assert "🚨" not in row["decision.summary"]
                assert "Alert" in row["decision.summary"]
                assert "detected" in row["decision.summary"]
                assert "🔴" not in row["decision.reasons"]
                assert "anomaly" in row["decision.reasons"]

    def test_write_frame_without_decision(self):
        """Test writing frame without decision data (fallback case)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "audit.csv"
            logger = DecisionAuditLogger(output_path)

            frame_data = {
                "unit_id": "unit_001",
                "timestamp": 1234567890.0,
                "cycle": 1,
                "state": "ALERT",
                "phase": "degradation",
                "structural_drift_score": 0.85,
                "composite_instability": 0.75,
                "decision": {},
            }

            logger.write_frame(frame_data)
            logger.close()

            # Verify the CSV was written with empty decision fields
            with open(output_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 1
                row = rows[0]
                assert row["unit_id"] == "unit_001"
                assert row["decision.severity"] == ""
                assert row["decision.summary"] == ""

    def test_write_multiple_frames(self):
        """Test writing multiple frames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "audit.csv"
            logger = DecisionAuditLogger(output_path)

            for i in range(5):
                frame_data = {
                    "unit_id": f"unit_{i:03d}",
                    "timestamp": 1234567890.0 + i,
                    "cycle": i,
                    "state": "STABLE" if i % 2 == 0 else "ALERT",
                    "phase": "normal" if i % 2 == 0 else "degradation",
                    "structural_drift_score": 0.2 + (i * 0.1),
                    "composite_instability": 0.15 + (i * 0.05),
                    "decision": {
                        "finding_confidence": 0.8 + (i * 0.02),
                        "action_confidence": 0.75 + (i * 0.03),
                        "transient_score": 0.05 * i,
                        "suppress": False,
                        "severity": "low" if i % 2 == 0 else "high",
                        "recommended_action": "monitor" if i % 2 == 0 else "inspect",
                        "recommended_target": "general",
                        "summary": f"Frame {i} status",
                        "reasons": ["reason1", "reason2"],
                    },
                }
                logger.write_frame(frame_data)

            logger.close()

            # Verify all frames were written
            with open(output_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 5
                for i, row in enumerate(rows):
                    assert row["unit_id"] == f"unit_{i:03d}"

    def test_context_manager(self):
        """Test using logger as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "audit.csv"

            with DecisionAuditLogger(output_path) as logger:
                frame_data = {
                    "unit_id": "unit_001",
                    "timestamp": 1234567890.0,
                    "cycle": 1,
                    "state": "ALERT",
                    "phase": "degradation",
                    "structural_drift_score": 0.85,
                    "composite_instability": 0.75,
                    "decision": {
                        "finding_confidence": 0.95,
                        "action_confidence": 0.90,
                        "transient_score": 0.1,
                        "suppress": False,
                        "severity": "high",
                        "recommended_action": "inspect",
                        "recommended_target": "bearing",
                        "summary": "Test frame",
                        "reasons": ["test"],
                    },
                }
                logger.write_frame(frame_data)

            # File should be closed and readable
            assert output_path.exists()
            with open(output_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 1

    def test_utf8_encoding(self):
        """Test that file is written with UTF-8 encoding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "audit.csv"
            logger = DecisionAuditLogger(output_path)

            frame_data = {
                "unit_id": "unit_001",
                "timestamp": 1234567890.0,
                "cycle": 1,
                "state": "ALERT",
                "phase": "degradation",
                "structural_drift_score": 0.85,
                "composite_instability": 0.75,
                "decision": {
                    "finding_confidence": 0.95,
                    "action_confidence": 0.90,
                    "transient_score": 0.1,
                    "suppress": False,
                    "severity": "high",
                    "recommended_action": "inspect",
                    "recommended_target": "bearing",
                    "summary": "Summary with special chars: café, naïve",
                    "reasons": ["reason1", "reason2"],
                },
            }

            logger.write_frame(frame_data)
            logger.close()

            # Verify file is readable with UTF-8 encoding
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
                assert "café" in content or "caf" in content
                assert len(content) > 0

    def test_list_field_handling(self):
        """Test that list fields (reasons) are properly joined."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "audit.csv"
            logger = DecisionAuditLogger(output_path)

            frame_data = {
                "unit_id": "unit_001",
                "timestamp": 1234567890.0,
                "cycle": 1,
                "state": "ALERT",
                "phase": "degradation",
                "structural_drift_score": 0.85,
                "composite_instability": 0.75,
                "decision": {
                    "finding_confidence": 0.95,
                    "action_confidence": 0.90,
                    "transient_score": 0.1,
                    "suppress": False,
                    "severity": "high",
                    "recommended_action": "inspect",
                    "recommended_target": "bearing",
                    "summary": "Test summary",
                    "reasons": ["Reason 1 🔴", "Reason 2 🟡", "Reason 3"],
                },
            }

            logger.write_frame(frame_data)
            logger.close()

            # Verify list fields are joined with semicolon and emoji removed
            with open(output_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 1
                row = rows[0]
                reasons = row["decision.reasons"]
                assert ";" in reasons  # Items should be semicolon-separated
                assert "🔴" not in reasons
                assert "🟡" not in reasons
                assert "Reason 1" in reasons
