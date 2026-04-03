from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import uuid4

from neraium_core.alignment import StructuralEngine
from neraium_core.grow.aggregation import aggregate_signals, latest_timestamp, records_by_scope
from neraium_core.grow.comparison import compare_entities
from neraium_core.grow.fleet import build_fleet_summary
from neraium_core.grow.forecast import estimate_time_to_impact_hours, humanize_impact_window
from neraium_core.grow.inefficiency import score_inefficiencies
from neraium_core.grow.interpretation import (
    biological_risk_label,
    dominant_relationship_shift,
    operator_message,
    urgency_for_state,
)
from neraium_core.grow.recommendations import recommended_action_for_driver
from neraium_core.grow.schemas import ComparisonSummary, DecisionRecord, EvidenceItem, GrowStateResponse, GrowTelemetryRecord, SetupSummary
from neraium_core.grow.scopes import SCOPE_ORDER, portfolio_scope_id, scope_id_for_record, scope_parent
from neraium_core.grow.validation import DEFAULT_THRESHOLDS, PLANT_MEASURED_SIGNALS


SIGNAL_GROUPS: dict[str, str] = {
    "zone_temperature": "climate_control",
    "zone_relative_humidity": "climate_control",
    "vapor_pressure_deficit": "climate_control",
    "co2_ppm": "co2_delivery",
    "airflow_rate": "airflow_management",
    "static_pressure": "airflow_management",
    "leaf_temperature": "biological_response",
    "canopy_temperature": "biological_response",
    "compressor_speed_rpm": "hvac_mechanical",
    "load_percent": "hvac_mechanical",
    "supply_air_temperature": "hvac_mechanical",
    "discharge_pressure": "hvac_mechanical",
    "filter_differential_pressure": "airflow_management",
    "irrigation_flow_rate": "irrigation_fertigation",
    "irrigation_pressure": "irrigation_fertigation",
    "nutrient_ec": "irrigation_fertigation",
    "drain_ec": "irrigation_fertigation",
    "fixture_power_kw": "lighting_photoperiod",
    "ppfd": "lighting_photoperiod",
    "room_power_kw": "power_load",
    "power_kw": "power_load",
    "transpiration_proxy": "biological_response",
    "stress_index": "biological_response",
    "image_health_score": "biological_response",
}

DRIVER_ALIASES = {
    "hvac_mechanical": "hvac_loop_inefficiency",
    "climate_control": "dehumidification_lag",
    "airflow_management": "airflow_restriction",
    "irrigation_fertigation": "irrigation_fertigation_drift",
    "lighting_photoperiod": "lighting_thermal_mismatch",
    "co2_delivery": "co2_response_instability",
    "power_load": "energy_recovery_mismatch",
    "biological_response": "biological_stress_escalation",
}

STATE_PRIORITY = {"STABLE": 0, "DRIFTING": 1, "AT_RISK": 2, "DEGRADING": 3}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class GrowRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            existing = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='grow_telemetry'").fetchone()
            if existing:
                cols = {row["name"] for row in conn.execute("PRAGMA table_info(grow_telemetry)").fetchall()}
                if "timestamp" not in cols:
                    conn.execute("DROP TABLE grow_telemetry")
            existing_decisions = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='grow_decisions'").fetchone()
            if existing_decisions:
                decision_cols = {row["name"] for row in conn.execute("PRAGMA table_info(grow_decisions)").fetchall()}
                if "payload_json" not in decision_cols or "entity_kind" in decision_cols:
                    conn.execute("DROP TABLE grow_decisions")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS grow_telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    facility_id TEXT NOT NULL,
                    site_id TEXT,
                    room_id TEXT,
                    zone_id TEXT,
                    plant_id TEXT,
                    asset_id TEXT,
                    subsystem_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS grow_decisions (
                    decision_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def add_record(self, record: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO grow_telemetry (timestamp, facility_id, site_id, room_id, zone_id, plant_id, asset_id, subsystem_id, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.get("timestamp"),
                    record.get("facility_id"),
                    record.get("site_id"),
                    record.get("room_id"),
                    record.get("zone_id"),
                    record.get("plant_id"),
                    record.get("asset_id"),
                    record.get("subsystem_id"),
                    json.dumps(record),
                ),
            )

    def all_records(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload_json FROM grow_telemetry ORDER BY timestamp ASC, id ASC").fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def add_decision(self, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO grow_decisions (decision_id, payload_json, created_at)
                VALUES (?, ?, ?)
                """,
                (payload["decision_id"], json.dumps(payload), _utc_now()),
            )

    def latest_decisions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM grow_decisions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def get_decision(self, decision_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM grow_decisions WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None


class GrowStateEngine:
    def __init__(self, db_path: str | Path, baseline_window: int = 8, recent_window: int = 4) -> None:
        self.repo = GrowRepository(db_path)
        self.structural_engines: dict[tuple[str, str, str], StructuralEngine] = {}
        self.baseline_window = baseline_window
        self.recent_window = recent_window

    def _engine(self, scope_type: str, facility_id: str, scope_id: str) -> StructuralEngine:
        key = (scope_type, facility_id, scope_id)
        if key not in self.structural_engines:
            self.structural_engines[key] = StructuralEngine(
                baseline_window=self.baseline_window,
                recent_window=self.recent_window,
            )
        return self.structural_engines[key]

    def ingest_records(self, records: list[GrowTelemetryRecord]) -> list[DecisionRecord]:
        touched: set[tuple[str, str]] = set()
        for record in records:
            payload = record.model_dump(mode="json")
            self.repo.add_record(payload)
            for scope in SCOPE_ORDER:
                scope_id = scope_id_for_record(scope, payload)
                if scope_id:
                    touched.add((scope, scope_id))
        decisions: list[DecisionRecord] = []
        for scope_type, scope_id in sorted(touched):
            state = self.get_state(scope_type, scope_id)
            if state is None:
                continue
            decision = self._build_decision(state)
            self.repo.add_decision(decision.model_dump(mode="json"))
            decisions.append(decision)
        return decisions[-20:]

    def _records_for_scope(self, scope_type: str, scope_id: str) -> list[dict[str, Any]]:
        grouped = records_by_scope(self.repo.all_records())
        return list(grouped.get(scope_type, {}).get(scope_id, []))

    def _baseline_and_recent(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if len(rows) <= self.baseline_window:
            return rows[:-1] or rows, rows[-self.recent_window :]
        return rows[: self.baseline_window], rows[-self.recent_window :]

    def _subsystem_scores(self, baseline: list[dict[str, Any]], recent: list[dict[str, Any]]) -> dict[str, float]:
        baseline_values: dict[str, list[float]] = defaultdict(list)
        recent_values: dict[str, list[float]] = defaultdict(list)
        for row in baseline:
            for signal, value in (row.get("signals") or {}).items():
                if isinstance(value, (int, float)):
                    baseline_values[signal].append(float(value))
        for row in recent:
            for signal, value in (row.get("signals") or {}).items():
                if isinstance(value, (int, float)):
                    recent_values[signal].append(float(value))
        scores: dict[str, float] = defaultdict(float)
        for signal, recent_series in recent_values.items():
            baseline_series = baseline_values.get(signal)
            if not baseline_series:
                continue
            delta = abs(mean(recent_series) - mean(baseline_series)) / max(abs(mean(baseline_series)), 1.0)
            scores[SIGNAL_GROUPS.get(signal, "cross_system_stability")] += delta

        latest_signals = recent[-1].get("signals", {}) if recent else {}
        if float(latest_signals.get("filter_differential_pressure") or 0.0) > 1.6:
            scores["airflow_management"] += 0.55
        if abs(float(latest_signals.get("drain_ec") or 0.0) - float(latest_signals.get("nutrient_ec") or 0.0)) > 0.6:
            scores["irrigation_fertigation"] += 0.6
        if float(latest_signals.get("zone_relative_humidity") or 0.0) > 65.0 and float(latest_signals.get("load_percent") or 0.0) < 45.0:
            scores["climate_control"] += 0.45
        if float(latest_signals.get("room_power_kw") or 0.0) > max(55.0, float(latest_signals.get("fixture_power_kw") or 0.0) * 2.6):
            scores["power_load"] += 0.4
        return dict(scores)

    def _threshold_status(self, signals: dict[str, Any]) -> tuple[bool, str]:
        breaches = []
        for signal, (low, high) in DEFAULT_THRESHOLDS.items():
            value = signals.get(signal)
            if isinstance(value, (int, float)) and not (low <= float(value) <= high):
                breaches.append(signal)
        if breaches:
            return True, f"Traditional thresholds were breached for: {', '.join(breaches[:4])}"
        return False, "No individual sensor exceeded thresholds at this point"

    def _trend_score(self, recent: list[dict[str, Any]]) -> float:
        if len(recent) < 2:
            return 0.0
        instabilities = []
        for row in recent:
            signals = row.get("signals", {})
            instabilities.append(float(signals.get("zone_temperature") or 0.0) + float(signals.get("stress_index") or 0.0) * 10.0)
        return round((instabilities[-1] - instabilities[0]) / max(len(instabilities) - 1, 1), 4)

    def _state_from_scores(self, drift_score: float, instability: float, slope: float, threshold_breach: bool) -> str:
        if threshold_breach or drift_score >= 0.62 or instability >= 0.68:
            return "DEGRADING"
        if drift_score >= 0.38 or instability >= 0.46 or slope >= 0.14:
            return "AT_RISK"
        if drift_score >= 0.18 or instability >= 0.22 or slope > 0.04:
            return "DRIFTING"
        return "STABLE"

    def _plant_mode(self, scope_type: str, rows: list[dict[str, Any]]) -> tuple[str, bool]:
        if scope_type != "plant":
            return "not_applicable", False
        measured = any(any(signal in (row.get("signals") or {}) for signal in PLANT_MEASURED_SIGNALS) for row in rows)
        return ("measured_plant_state" if measured else "inferred_plant_risk_view"), measured

    def _comparison_summary(self, scope_type: str, state: dict[str, Any]) -> ComparisonSummary:
        peers = [peer for peer in self.list_states(scope_type, include_comparison=False) if peer.get("scope_id") != state.get("scope_id")]
        comparable = [state, *peers] if peers else [state]
        return ComparisonSummary(**compare_entities(state, comparable))

    def _context_fields(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        latest = rows[-1]
        return {
            "facility_id": latest.get("facility_id"),
            "site_id": latest.get("site_id"),
            "room_id": latest.get("room_id"),
            "zone_id": latest.get("zone_id"),
            "plant_id": latest.get("plant_id"),
            "asset_id": latest.get("asset_id"),
            "subsystem": latest.get("subsystem_id"),
        }

    def get_state(self, scope_type: str, scope_id: str, include_comparison: bool = True) -> GrowStateResponse | None:
        rows = self._records_for_scope(scope_type, scope_id)
        if not rows:
            return None
        rows = sorted(rows, key=lambda row: row["timestamp"])
        baseline, recent = self._baseline_and_recent(rows)
        latest = recent[-1]
        aggregated_signals = aggregate_signals(recent)
        subsystem_scores = self._subsystem_scores(baseline, recent)
        driver_group = max(subsystem_scores, key=subsystem_scores.get) if subsystem_scores else "climate_control"
        driver_key = DRIVER_ALIASES.get(driver_group, "hvac_loop_inefficiency")
        shift_text, cross_driver_text = dominant_relationship_shift({driver_key: subsystem_scores.get(driver_group, 0.0)}, aggregated_signals)

        engine_input = {
            "timestamp": latest_timestamp(recent),
            "customer_id": latest.get("facility_id"),
            "site_id": scope_id,
            "asset_id": scope_id,
            "sensor_values": {key: value for key, value in aggregated_signals.items() if isinstance(value, (int, float))},
        }
        try:
            structural = self._engine(scope_type, latest["facility_id"], scope_id).process_frame(engine_input)
            drift_score = round(float(structural.get("structural_drift_score", structural.get("latest_drift", 0.0)) or 0.0), 3)
        except Exception:
            structural = {}
            drift_score = round(sum(subsystem_scores.values()) / max(len(subsystem_scores), 1), 3) if subsystem_scores else 0.0
        ineff = score_inefficiencies(aggregated_signals, subsystem_scores)
        composite_instability = round(
            max(
                drift_score,
                sum(float(ineff[key]) for key in ("energy_inefficiency", "climate_inefficiency", "process_inefficiency", "biological_inefficiency")) / 4.0,
                sum(subsystem_scores.values()) / max(len(subsystem_scores), 1) if subsystem_scores else 0.0,
            ),
            3,
        )
        threshold_breach_detected, threshold_message = self._threshold_status(aggregated_signals)
        trend_score = max(0.0, self._trend_score(recent))
        state = self._state_from_scores(drift_score, composite_instability, trend_score, threshold_breach_detected)
        impact_hours = estimate_time_to_impact_hours(
            drift_score=drift_score,
            instability=composite_instability,
            slope=trend_score,
            threshold_breach_detected=threshold_breach_detected,
        )
        urgency = urgency_for_state(state, impact_hours)
        plant_mode, plant_measured = self._plant_mode(scope_type, rows)
        biological_risk = biological_risk_label(state=state, signals=aggregated_signals, threshold_breach_detected=threshold_breach_detected).replace(" ", "_")
        confidence = round(min(0.98, 0.42 + min(1.0, len(rows) / max(self.baseline_window + self.recent_window, 1)) * 0.32 + min(composite_instability, 0.24)), 2)
        context = self._context_fields(rows)
        comparison_summary = (
            self._comparison_summary(scope_type, {
                "scope_id": scope_id,
                "structural_drift_score": drift_score,
                "composite_instability": composite_instability,
                **ineff,
                "cross_system_driver": cross_driver_text,
                "dominant_relationship_shift": shift_text,
            })
            if include_comparison
            else ComparisonSummary(
                baseline_delta=round(drift_score, 3),
                peer_median_delta=0.0,
                best_performer_delta=0.0,
                ranking_position=1,
                peer_count=1,
                relative_drift=0.0,
                relative_inefficiency=0.0,
                likely_causal_differences=[],
            )
        )
        parent_scope_id = scope_parent(scope_type, latest)
        state_start = baseline[-1]["timestamp"] if baseline else latest["timestamp"]
        time_in_state = int((datetime.fromisoformat(str(latest["timestamp"]).replace("Z", "+00:00")) - datetime.fromisoformat(str(state_start).replace("Z", "+00:00"))).total_seconds() / 60)
        evidence = [
            EvidenceItem(summary=f"{scope_type.title()} structural drift score is {drift_score:.2f} with composite instability {composite_instability:.2f}", subsystem=driver_group),
            EvidenceItem(summary=threshold_message, severity="warning" if threshold_breach_detected else "info", subsystem=driver_group),
            EvidenceItem(summary=ineff["inefficiency_summary"], subsystem=driver_group),
        ]
        if scope_type == "plant" and not plant_measured:
            evidence.append(EvidenceItem(summary="Plant view is inferred from local microclimate and zone context because direct plant telemetry is limited", subsystem=driver_group))
        return GrowStateResponse(
            timestamp=latest_timestamp(recent),
            scope_type=scope_type,  # type: ignore[arg-type]
            scope_id=scope_id,
            parent_scope_id=parent_scope_id,
            facility_id=context["facility_id"],
            site_id=context["site_id"],
            room_id=context["room_id"],
            zone_id=context["zone_id"],
            plant_id=context["plant_id"],
            asset_id=context["asset_id"],
            subsystem=context["subsystem"],
            state=state,  # type: ignore[arg-type]
            confidence=confidence,
            structural_drift_score=drift_score,
            composite_instability=composite_instability,
            time_in_state_minutes=max(time_in_state, 0),
            trend_direction=("WORSENING" if trend_score > 0.05 else "STABLE"),  # type: ignore[arg-type]
            dominant_relationship_shift=shift_text,
            cross_system_driver=cross_driver_text,
            biological_risk=biological_risk,  # type: ignore[arg-type]
            inefficiency_summary=str(ineff["inefficiency_summary"]),
            energy_inefficiency=float(ineff["energy_inefficiency"]),
            climate_inefficiency=float(ineff["climate_inefficiency"]),
            process_inefficiency=float(ineff["process_inefficiency"]),
            biological_inefficiency=float(ineff["biological_inefficiency"]),
            recommended_action=recommended_action_for_driver(driver_key),
            urgency=urgency,  # type: ignore[arg-type]
            estimated_time_to_impact_hours=impact_hours,
            threshold_breach_detected=threshold_breach_detected,
            threshold_breach_message=threshold_message,
            operator_message=operator_message(
                state=state,
                relationship_shift=shift_text,
                cross_system_driver=cross_driver_text,
                impact_hours=impact_hours,
                threshold_text=threshold_message,
                driver_key=driver_key,
            ),
            evidence=evidence,
            comparison_summary=comparison_summary,
            setup_summary=self.setup_summary(),
            plant_view_mode=plant_mode,  # type: ignore[arg-type]
            measured_plant_signals_present=plant_measured,
            trend_score=trend_score,
        )

    def list_states(self, scope_type: str, include_comparison: bool = True) -> list[dict[str, Any]]:
        grouped = records_by_scope(self.repo.all_records()).get(scope_type, {})
        states = []
        for scope_id in grouped:
            state = self.get_state(scope_type, scope_id, include_comparison=include_comparison)
            if state:
                states.append(state.model_dump(mode="json"))
        return sorted(states, key=lambda row: float(row.get("composite_instability", 0.0)), reverse=True)

    def timeline(self, scope_type: str, scope_id: str) -> list[dict[str, Any]]:
        rows = self._records_for_scope(scope_type, scope_id)
        if not rows:
            return []
        ordered = sorted(rows, key=lambda row: row["timestamp"])
        out = []
        for end in range(1, len(ordered) + 1):
            partial_engine = GrowStateEngine(db_path=":memory:", baseline_window=self.baseline_window, recent_window=self.recent_window)
            for row in ordered[:end]:
                partial_engine.repo.add_record(row)
            state = partial_engine.get_state(scope_type, scope_id)
            if state:
                out.append({
                    "timestamp": state.timestamp,
                    "state": state.state,
                    "structural_drift_score": state.structural_drift_score,
                    "confidence": state.confidence,
                    "dominant_relationship_shift": state.dominant_relationship_shift,
                    "cross_system_driver": state.cross_system_driver,
                    "energy_inefficiency": state.energy_inefficiency,
                    "climate_inefficiency": state.climate_inefficiency,
                    "process_inefficiency": state.process_inefficiency,
                    "biological_inefficiency": state.biological_inefficiency,
                })
        return out

    def _build_decision(self, state: GrowStateResponse) -> DecisionRecord:
        return DecisionRecord(
            decision_id=f"{state.scope_type}-{state.scope_id}-{uuid4().hex[:8]}",
            timestamp=state.timestamp,
            scope_type=state.scope_type,
            scope_id=state.scope_id,
            state=state.state,
            confidence=state.confidence,
            time_in_state=f"{state.time_in_state_minutes} minutes",
            trend_direction=state.trend_direction,
            dominant_relationship_shift=state.dominant_relationship_shift,
            cross_system_driver=state.cross_system_driver,
            biological_risk=state.biological_risk,
            inefficiency_summary=state.inefficiency_summary,
            recommended_action=state.recommended_action,
            urgency=state.urgency,
            estimated_time_to_impact=humanize_impact_window(state.estimated_time_to_impact_hours),
            evidence_summary=[item.summary for item in state.evidence],
            threshold_breach_detected=state.threshold_breach_detected,
            threshold_breach_message=state.threshold_breach_message,
        )

    def latest_decisions(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.repo.latest_decisions(limit)

    def get_decision(self, decision_id: str) -> dict[str, Any] | None:
        return self.repo.get_decision(decision_id)

    def compare_entities(self, scope_type: str, scope_ids: list[str]) -> dict[str, Any]:
        states = [self.get_state(scope_type, scope_id) for scope_id in scope_ids]
        kept = [state for state in states if state is not None]
        if not kept:
            return {"anchor": None, "peers": [], "summary": {}}
        anchor = kept[0]
        peers = [state.model_dump(mode="json") for state in kept[1:]]
        summary = compare_entities(anchor.model_dump(mode="json"), [state.model_dump(mode="json") for state in kept])
        return {"anchor": anchor.model_dump(mode="json"), "peers": peers, "summary": summary}

    def compare_scope(self, scope_type: str, scope_id: str | None = None) -> dict[str, Any]:
        states = self.list_states(scope_type)
        if scope_id:
            ordered = [state for state in states if state.get("scope_id") == scope_id] + [state for state in states if state.get("scope_id") != scope_id]
        else:
            ordered = states
        if not ordered:
            return {"anchor": None, "peers": [], "summary": {}}
        return {"anchor": ordered[0], "peers": ordered[1:6], "summary": compare_entities(ordered[0], ordered[:6])}

    def setup_summary(self) -> SetupSummary:
        records = self.repo.all_records()
        grouped = records_by_scope(records)
        signals = set()
        sources = set()
        sites = set()
        for row in records:
            signals.update((row.get("signals") or {}).keys())
            if row.get("telemetry_source"):
                sources.add(str(row["telemetry_source"]))
            if row.get("site_id"):
                sites.add(str(row["site_id"]))
        readiness = min(100, int(len(records) * 4 + len(signals) / 2))
        return SetupSummary(
            deployment_mode="grow",
            telemetry_source_count=max(len(sources), 1 if records else 0),
            signal_count_discovered=len(signals),
            rooms_discovered=len(grouped["room"]),
            zones_discovered=len(grouped["zone"]),
            plants_discovered=len(grouped["plant"]),
            assets_discovered=len({row.get("asset_id") for row in records if row.get("asset_id")}),
            subsystems_discovered=len(grouped["subsystem"]),
            facilities_discovered=len(grouped["facility"]),
            portfolio_sites_discovered=len(sites),
            baseline_learned=bool(records and len(records) >= self.baseline_window),
            model_ready=bool(records and len(records) >= self.baseline_window),
            readiness_percent=readiness,
            estimated_time_to_readiness_minutes=max(0, 60 - min(60, len(records) * 2)),
            last_ingest_timestamp=records[-1]["timestamp"] if records else None,
        )

    def facility_summary(self) -> dict[str, Any]:
        facilities = self.list_states("facility")
        headline = "Neraium scales from single-plant risk to portfolio-wide operational intelligence - showing where the grow is drifting, why, and where inefficiencies are spreading before yield is impacted."
        return {
            "headline": headline,
            "state": facilities[0]["state"] if facilities else "STABLE",
            "facilities": facilities,
            "top_drivers": [item["driver"] for item in self.fleet_summary()["top_cross_system_drivers"][:4]],
            "supporting_copy": [
                "Read-only and vendor-neutral",
                "Hidden drift and inefficiency surfaced before threshold breach",
                "Measured and inferred plant views supported",
            ],
            "setup_summary": self.setup_summary().model_dump(mode="json"),
        }

    def portfolio_summary(self) -> dict[str, Any]:
        portfolio_state = self.get_state("portfolio", portfolio_scope_id())
        return {
            "portfolio": portfolio_state.model_dump(mode="json") if portfolio_state else None,
            "facilities": self.list_states("facility"),
            "highest_inefficiency_areas": self.fleet_summary()["highest_inefficiency_areas"],
        }

    def fleet_summary(self) -> dict[str, Any]:
        return build_fleet_summary(
            self.list_states("plant"),
            self.list_states("zone"),
            self.list_states("room"),
            self.list_states("subsystem"),
            self.list_states("facility"),
        )
