"""Day 11: alert detection, logging, suppression, and operator inbox."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from config import DATE_COLUMN

_DEFAULT_ALERT_LOG = Path("output") / "alert_log.jsonl"

_WARNING_ORDER = {"none": 0, "low": 1, "moderate": 2, "elevated": 3, "high": 4}
_SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3}

_DEFENSIVE = frozenset({"reduce_risk", "defensive", "wait", "neutral"})


def _wl_rank(w: str) -> int:
    return _WARNING_ORDER.get(str(w).lower(), 0)


def _sev_rank(s: str) -> int:
    return _SEVERITY_ORDER.get(str(s).lower(), 0)


def detect_alert_conditions(
    current_snapshot: dict[str, Any],
    prior_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Compare snapshots and decide if an alert should fire.

    Returns ``alert_triggered``, ``alert_type``, ``alert_severity``, ``alert_message``, ``reason_codes``.
    """
    if not current_snapshot:
        return {
            "alert_triggered": False,
            "alert_type": "none",
            "alert_severity": "info",
            "alert_message": "No snapshot",
            "reason_codes": [],
        }

    reasons: list[str] = []
    if prior_snapshot is None:
        return {
            "alert_triggered": False,
            "alert_type": "none",
            "alert_severity": "info",
            "alert_message": "No prior snapshot; skipping alert until next cycle.",
            "reason_codes": ["no_prior"],
        }

    cr = str(current_snapshot.get("market_regime_label", ""))
    pr = str(prior_snapshot.get("market_regime_label", ""))
    if cr != pr:
        reasons.append("regime_change")
    cwl = str(current_snapshot.get("market_warning_level", "none"))
    pwl = str(prior_snapshot.get("market_warning_level", "none"))
    if _wl_rank(cwl) >= _wl_rank(pwl) + 2 or (_wl_rank(cwl) > _wl_rank(pwl) and cwl not in ("none", "low")):
        reasons.append("warning_escalation")

    cact = str(current_snapshot.get("path_adjusted_market_action", ""))
    pact = str(prior_snapshot.get("path_adjusted_market_action", ""))
    if cact != pact:
        reasons.append("action_shift")

    try:
        cc = float(current_snapshot.get("market_confidence_score", float("nan")))
        pc = float(prior_snapshot.get("market_confidence_score", float("nan")))
        if not pd.isna(cc) and not pd.isna(pc):
            if pc >= 0.55 and cc < 0.45:
                reasons.append("confidence_break")
    except (TypeError, ValueError):
        pass

    if cact in _DEFENSIVE and pact not in _DEFENSIVE:
        reasons.append("path_deterioration")

    scen = str(current_snapshot.get("scenario_path_label", ""))
    if scen in {"fragile_to_risk_off", "false_calm_break"}:
        reasons.append("high_risk_path")

    if not reasons:
        return {
            "alert_triggered": False,
            "alert_type": "none",
            "alert_severity": "info",
            "alert_message": "No material change vs prior snapshot.",
            "reason_codes": [],
        }

    # severity
    if "high_risk_path" in reasons or "warning_escalation" in reasons and _wl_rank(cwl) >= 3:
        sev = "high"
    elif "regime_change" in reasons or "path_deterioration" in reasons:
        sev = "medium"
    elif "confidence_break" in reasons:
        sev = "medium"
    else:
        sev = "low"

    # primary type
    if "regime_change" in reasons:
        atype = "regime_change"
    elif "warning_escalation" in reasons:
        atype = "warning_escalation"
    elif "path_deterioration" in reasons:
        atype = "path_deterioration"
    elif "confidence_break" in reasons:
        atype = "confidence_break"
    elif "action_shift" in reasons:
        atype = "action_shift"
    elif "high_risk_path" in reasons:
        atype = "path_deterioration"
    else:
        atype = "action_shift"

    msg_parts = [f"{atype}: {', '.join(reasons)}"]
    msg_parts.append(f"regime {pr}->{cr}" if pr != cr else f"regime {cr}")
    msg_parts.append(f"warning {pwl}->{cwl}")
    msg_parts.append(f"action {pact}->{cact}")

    return {
        "alert_triggered": True,
        "alert_type": atype,
        "alert_severity": sev,
        "alert_message": "; ".join(msg_parts),
        "reason_codes": reasons,
    }


def build_alert_record(
    current_snapshot: dict[str, Any],
    alert_info: dict[str, Any],
) -> dict[str, Any]:
    """Structured operator payload."""
    aid = str(uuid.uuid4())
    ts = current_snapshot.get("timestamp", "")
    return {
        "alert_id": aid,
        "timestamp": ts,
        "alert_type": alert_info.get("alert_type", "none"),
        "alert_severity": alert_info.get("alert_severity", "info"),
        "market_regime_label": current_snapshot.get("market_regime_label", ""),
        "market_warning_level": current_snapshot.get("market_warning_level", ""),
        "path_adjusted_market_action": current_snapshot.get("path_adjusted_market_action", ""),
        "market_confidence_score": float(current_snapshot.get("market_confidence_score", float("nan"))),
        "alert_message": alert_info.get("alert_message", ""),
        "reason_codes": list(alert_info.get("reason_codes", [])),
    }


def log_alert_record(
    alert_record: dict[str, Any],
    output_path: str | Path = _DEFAULT_ALERT_LOG,
) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(alert_record, default=str, ensure_ascii=False) + "\n")


def load_recent_alerts(
    output_path: str | Path = _DEFAULT_ALERT_LOG,
    limit: int = 20,
) -> list[dict[str, Any]]:
    p = Path(output_path)
    if not p.is_file():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    tail = lines[-limit:] if limit > 0 else lines
    out: list[dict[str, Any]] = []
    for line in tail:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def should_suppress_alert(
    current_alert: dict[str, Any],
    recent_alerts: list[dict[str, Any]],
    *,
    window_seconds: float = 3600.0,
) -> bool:
    """
    Deterministic suppression: repeated same type+severity+regime within window,
    or low severity after recent high for same type.
    """
    if not recent_alerts:
        return False
    at = str(current_alert.get("alert_type", ""))
    sev = str(current_alert.get("alert_severity", ""))
    reg = str(current_alert.get("market_regime_label", ""))
    cur_ts = str(current_alert.get("timestamp", ""))

    for prev in reversed(recent_alerts[-10:]):
        if str(prev.get("alert_type")) == at and str(prev.get("alert_severity")) == sev and str(prev.get("market_regime_label")) == reg:
            # same fingerprint — suppress repeat
            return True
        if at == str(prev.get("alert_type")) and _sev_rank(sev) < _sev_rank(str(prev.get("alert_severity", "info"))):
            return True
    return False


def build_operator_inbox(
    signals_df: pd.DataFrame,
    alert_log_path: str | Path = _DEFAULT_ALERT_LOG,
    *,
    tail_rows: int = 50,
) -> pd.DataFrame:
    """
    Merge recent signal rows with alert metadata (last alert per timestamp match best-effort).

    ``signals_df`` is typically ``market_state`` (must include Day 9–10 columns).
    Sorted **newest first** (descending timestamp).
    """
    cols = [
        DATE_COLUMN,
        "market_regime_label",
        "market_warning_level",
        "path_adjusted_market_action",
        "market_confidence_score",
        "alert_type",
        "alert_severity",
        "alert_message",
        "explanation",
    ]
    if signals_df.empty:
        return pd.DataFrame(columns=cols)

    sub = signals_df.sort_values(DATE_COLUMN, ascending=False).head(tail_rows).copy()

    alerts = load_recent_alerts(alert_log_path, limit=100)
    alert_by_ts: dict[str, dict[str, Any]] = {}
    for a in alerts:
        alert_by_ts[str(a.get("timestamp", ""))] = a

    rows: list[dict[str, Any]] = []
    for _, row in sub.iterrows():
        ts = row[DATE_COLUMN]
        ts_key = pd.Timestamp(ts).isoformat()
        am = alert_by_ts.get(ts_key, {})
        expl = str(row.get("path_aware_market_explanation", row.get("market_explanation", "")))
        rows.append(
            {
                DATE_COLUMN: ts,
                "market_regime_label": row.get("market_regime_label", ""),
                "market_warning_level": row.get("market_warning_level", ""),
                "path_adjusted_market_action": row.get("path_adjusted_market_action", ""),
                "market_confidence_score": float(row.get("market_confidence_score", float("nan"))),
                "alert_type": am.get("alert_type", ""),
                "alert_severity": am.get("alert_severity", ""),
                "alert_message": am.get("alert_message", ""),
                "explanation": expl,
            }
        )

    inbox = pd.DataFrame(rows)
    return inbox.sort_values(DATE_COLUMN, ascending=False).reset_index(drop=True)
