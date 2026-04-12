"""Structural drift detection and state machine.

Responsibilities:
- Compute drift score from correlation matrix changes
- EMA smoothing of drift signal
- Calibrate watch/alert thresholds from baseline samples
- Drift state machine: STABLE → WATCH → ALERT
- Alert persistence and unlatching logic
"""

# Stub: implementation will be extracted from alignment.py in Phase 3
