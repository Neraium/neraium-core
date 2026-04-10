from __future__ import annotations

import json
import logging
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .state_store import RuntimeStateStore

DEMO_STRUCTURAL_SENSOR_KEYS = [
    "pressure",
    "flow",
    "vibration",
    "temperature",
    "motor_current",
    "bearing_temp",
    "load_cell",
    "rpm",
    "humidity",
    "displacement",
    "valve_position",
    "shaft_accel",
    "lubrication_psi",
    "seismic_x",
    "seismic_y",
    "winding_temp",
    "inlet_guide",
    "outlet_guide",
    "torque_est",
    "casing_vibe",
    "oil_quality",
    "stator_temp",
    "field_bus_ok",
    "coolant_flow",
]


class DemoJobsManager:
    def __init__(
        self,
        *,
        store: RuntimeStateStore,
        service_instance: Any,
        resolve_customer_id: Any,
        utc_now_iso: Any,
        log_structured: Any,
        summarize_exception_for_logs: Any,
        logger: logging.Logger,
    ) -> None:
        self.store = store
        self._service_instance = service_instance
        self._resolve_customer_id = resolve_customer_id
        self._utc_now_iso = utc_now_iso
        self._log_structured = log_structured
        self._summarize_exception_for_logs = summarize_exception_for_logs
        self._logger = logger

    def public_demo_job(self, job: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": str(job.get("job_id")),
            "status": str(job.get("status", "unknown")),
            "run_id": str(job.get("run_id") or ""),
            "customer_id": self._resolve_customer_id(job.get("customer_id")),
            "progress": max(0, min(100, int(job.get("progress", 0)))),
            "processed": max(0, int(job.get("processed", 0))),
            "total_frames": max(0, int(job.get("total_frames", 0))),
            "message": str(job.get("message") or ""),
            "error": job.get("error"),
            "created_at": str(job.get("created_at") or self._utc_now_iso()),
            "updated_at": str(job.get("updated_at") or self._utc_now_iso()),
        }

    def update_demo_job(self, job_id: str, **fields: Any) -> dict[str, Any] | None:
        with self.store.demo_jobs_lock:
            job = self.store.demo_jobs.get(job_id)
            if job is None:
                return None
            job.update(fields)
            job["updated_at"] = self._utc_now_iso()
            return dict(job)

    def start_demo_seed_job(self, *, job_id: str, resolved_run: str, resolved_customer: str, payload: Any) -> None:
        threading.Thread(
            target=self.run_demo_seed_job,
            kwargs={
                "job_id": job_id,
                "resolved_run": resolved_run,
                "resolved_customer": resolved_customer,
                "payload": payload,
            },
            daemon=True,
        ).start()

    def run_demo_seed_job(self, *, job_id: str, resolved_run: str, resolved_customer: str, payload: Any) -> None:
        minutes = int(payload.minutes)
        total = max(10, min(240, minutes))
        now = datetime.now(timezone.utc)
        processed = 0
        failure_frame = None
        self.update_demo_job(
            job_id,
            status="running",
            message="Seeding telemetry on server...",
            total_frames=total,
            progress=0,
            processed=0,
            run_id=resolved_run,
            customer_id=resolved_customer,
        )
        self._log_structured(
            self._logger,
            event="demo_seed_start",
            fields={
                "job_id": job_id,
                "run_id": resolved_run,
                "customer_id": resolved_customer,
                "total_frames": total,
                "profile": payload.profile,
            },
        )
        try:
            for i in range(total):
                failure_frame = i + 1
                timestamp = (now.replace(microsecond=0).timestamp() - (total - i) * 60.0)
                t = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
                if payload.profile == "watch":
                    drift_lift = 0.35 + (i / max(1, total - 1)) * 0.35
                    vib_spike = 0.55 + (i / max(1, total - 1)) * 0.45
                elif payload.profile == "critical":
                    drift_lift = 0.25 + (i / max(1, total - 1)) * 0.85
                    vib_spike = 0.8 + (i / max(1, total - 1)) * 1.8
                elif payload.profile == "sample":
                    drift_lift = 0.2 if i < total // 3 else 0.6 if i < (2 * total) // 3 else 1.0
                    vib_spike = drift_lift * 1.1
                else:
                    drift_lift = 0.12
                    vib_spike = 0.2
                p = i / max(1, total - 1)
                frame = {
                    "timestamp": t,
                    "site_id": payload.site_id,
                    "asset_id": payload.asset_id,
                    "sensor_values": self._build_demo_sensor_values_row(i, p, drift_lift, vib_spike),
                    "customer_id": resolved_customer,
                }
                self._service_instance.ingest_frame(
                    frame,
                    run_id=resolved_run,
                    customer_id=resolved_customer,
                )
                processed += 1
                if processed % 10 == 0 or processed == total:
                    progress = int((processed / max(1, total)) * 100)
                    self.update_demo_job(
                        job_id,
                        progress=progress,
                        processed=processed,
                        message=f"Seeding telemetry on server... ({processed}/{total})",
                    )
                    self._log_structured(
                        self._logger,
                        event="demo_seed_progress",
                        fields={
                            "job_id": job_id,
                            "run_id": resolved_run,
                            "customer_id": resolved_customer,
                            "processed": processed,
                            "total_frames": total,
                            "progress": progress,
                        },
                    )
        except Exception as exc:
            detail = self._summarize_exception_for_logs(exc)
            self.update_demo_job(
                job_id,
                status="error",
                progress=int((processed / max(1, total)) * 100),
                processed=processed,
                message="Demo seed failed.",
                error=detail,
            )
            self._log_structured(
                self._logger,
                event="demo_seed_failure",
                fields={
                    "job_id": job_id,
                    "run_id": resolved_run,
                    "customer_id": resolved_customer,
                    "processed": processed,
                    "total_frames": total,
                    "failure_frame": failure_frame,
                    "error": detail,
                },
                level=logging.ERROR,
            )
            return
        self.update_demo_job(
            job_id,
            status="complete",
            progress=100,
            processed=processed,
            message="Demo seeded successfully",
            error=None,
        )
        self._log_structured(
            self._logger,
            event="demo_seed_complete",
            fields={
                "job_id": job_id,
                "run_id": resolved_run,
                "customer_id": resolved_customer,
                "processed": processed,
                "total_frames": total,
            },
        )

    def load_greenhouse_demo_subset(self, max_frames: int) -> list[dict[str, Any]]:
        limited = max(30, min(500, int(max_frames)))
        with self.store.greenhouse_demo_cache_lock:
            cached = self.store.greenhouse_demo_cache.get(limited)
            if cached is not None:
                return list(cached)

        scenario_path = Path(__file__).resolve().parents[1] / "demo_data" / "cannabis_grow_op_scenario.json"
        if not scenario_path.is_file():
            raise FileNotFoundError(f"Greenhouse scenario file missing at {scenario_path}")

        payload = scenario_path.read_text(encoding="utf-8")
        raw = json.loads(payload)
        asset = raw.get("asset") or {}
        site_id = str(asset.get("site_id") or "grow-op-facility-01")
        asset_id = str(asset.get("asset_id") or "canopy-zone-A")

        scenario_rows: list[dict[str, Any]] = []
        for phase in raw.get("phases") or []:
            for frame in phase.get("frames") or []:
                sensor_values = frame.get("sensor_values")
                if not isinstance(sensor_values, dict):
                    continue
                offset = int(frame.get("minute_offset", 0))
                scenario_rows.append(
                    {
                        "minute_offset": offset,
                        "sensor_values": {k: float(v) for k, v in sensor_values.items()},
                    }
                )

        if not scenario_rows:
            raise ValueError("Greenhouse scenario is empty.")

        scenario_rows.sort(key=lambda row: int(row.get("minute_offset", 0)))
        selected = scenario_rows[:limited]
        now = datetime.now(timezone.utc).replace(microsecond=0)
        rows: list[dict[str, Any]] = []
        for idx, row in enumerate(selected):
            timestamp = now.timestamp() - max(0, limited - idx) * 60.0
            rows.append(
                {
                    "timestamp": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                    "site_id": site_id,
                    "asset_id": asset_id,
                    "sensor_values": dict(row["sensor_values"]),
                }
            )

        with self.store.greenhouse_demo_cache_lock:
            self.store.greenhouse_demo_cache[limited] = list(rows)
        return rows

    @staticmethod
    def _build_demo_sensor_values_row(i: int, p: float, drift_lift: float, vib_spike: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, key in enumerate(DEMO_STRUCTURAL_SENSOR_KEYS):
            phase = k * 0.85
            wave = math.sin(i / (5.2 + k * 0.11) + phase)
            w2 = math.cos(i / (7.1 + k * 0.09) + phase * 0.65)
            base = 18 + k * 6.2
            out[key] = (
                base
                + wave * (1 + drift_lift * (0.45 + k * 0.025))
                + w2 * (0.55 + vib_spike * 0.12)
                + i * (0.011 + k * 0.0008)
                + p * (0.15 + k * 0.02)
            )
        return out
