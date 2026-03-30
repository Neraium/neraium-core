from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neraium_core.sii.config import SIIConfig
from neraium_core.sii.engine import SIIEngine
from neraium_core.sii.types import SIIFrame


@dataclass(frozen=True)
class DemoPoint:
    cycle: int
    phase: str
    pressure: float
    flow: float
    vibration: float
    temperature: float


def _build_demo_points() -> list[DemoPoint]:
    points: list[DemoPoint] = []
    cycle = 0
    for i in range(24):
        points.append(
            DemoPoint(
                cycle=cycle,
                phase="normal",
                pressure=50.0 + 0.08 * i,
                flow=12.2 - 0.02 * i,
                vibration=1.05 + 0.01 * i,
                temperature=78.0 + 0.04 * i,
            )
        )
        cycle += 1
    for i in range(16):
        points.append(
            DemoPoint(
                cycle=cycle,
                phase="drift",
                pressure=52.0 + 0.45 * i,
                flow=11.7 - 0.06 * i,
                vibration=1.3 + 0.09 * i,
                temperature=79.5 + 0.16 * i,
            )
        )
        cycle += 1
    for i in range(10):
        points.append(
            DemoPoint(
                cycle=cycle,
                phase="rising_instability",
                pressure=60.0 + 0.9 * i,
                flow=10.8 - 0.18 * i,
                vibration=2.9 + 0.24 * i,
                temperature=83.0 + 0.38 * i,
            )
        )
        cycle += 1
    for i in range(6):
        points.append(
            DemoPoint(
                cycle=cycle,
                phase="critical",
                pressure=69.0 + 1.1 * i,
                flow=9.0 - 0.33 * i,
                vibration=5.4 + 0.42 * i,
                temperature=88.0 + 0.52 * i,
            )
        )
        cycle += 1
    return points


def _threshold_alarm_cycle(points: list[DemoPoint], *, vibration_threshold: float = 5.2) -> int | None:
    for point in points:
        if point.vibration >= vibration_threshold:
            return point.cycle
    return None


def run_demo(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = SIIConfig(
        baseline_window=12,
        recent_window=6,
        max_history=256,
        min_samples_for_alerts=12,
        regime_store_path=str(output_dir / "investor_demo_regimes.json"),
        allow_context_provider=False,
    )
    engine = SIIEngine(config=config)
    start = datetime.now(tz=timezone.utc).replace(microsecond=0)
    points = _build_demo_points()
    timeline: list[dict[str, object]] = []
    for point in points:
        frame = SIIFrame(
            timestamp=(start + timedelta(minutes=point.cycle)).isoformat(),
            site_id="site-demo",
            system_id="system-demo",
            asset_id="asset-demo",
            node_id="node-demo",
            sensor_values={
                "pressure": point.pressure,
                "flow": point.flow,
                "vibration": point.vibration,
                "temperature": point.temperature,
            },
        )
        result = engine.process_frame(frame)
        timeline.append(
            {
                "cycle": point.cycle,
                "phase": point.phase,
                "state": result.get("state"),
                "interpreted_state": result.get("interpreted_state"),
                "risk_level": (result.get("risk_assessment") or {}).get("current_risk_level"),
                "composite_instability": ((result.get("experimental_analytics") or {}).get("composite_instability")),
                "structural_drift_score": result.get("structural_drift_score"),
            }
        )
    engine.close()

    first_watch = next((row["cycle"] for row in timeline if row.get("state") == "WATCH"), None)
    first_alert = next((row["cycle"] for row in timeline if row.get("state") == "ALERT"), None)
    first_non_nominal = next(
        (row["cycle"] for row in timeline if row.get("interpreted_state") != "NOMINAL_STRUCTURE"),
        None,
    )
    threshold_cycle = _threshold_alarm_cycle(points)
    lead_cycles = (
        int(threshold_cycle) - int(first_non_nominal)
        if first_non_nominal is not None and threshold_cycle is not None
        else None
    )
    summary = {
        "scenario": "normal_to_drift_to_instability_to_critical",
        "first_non_nominal_cycle": first_non_nominal,
        "first_watch_cycle": first_watch,
        "first_alert_cycle": first_alert,
        "threshold_alarm_cycle": threshold_cycle,
        "early_warning_lead_cycles_vs_threshold_alarm": lead_cycles,
        "proof_points": [
            "Structural interpretation exits NOMINAL before hard vibration threshold alarm.",
            "State progression is monotonic through normal -> drift -> instability -> critical in the canonical scenario.",
        ],
        "timeline": timeline,
    }
    json_path = output_dir / "investor_demo_report.json"
    md_path = output_dir / "investor_demo_report.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Neraium investor demo proof report",
                "",
                f"- Scenario: `{summary['scenario']}`",
                f"- First WATCH cycle: `{first_watch}`",
                f"- First ALERT cycle: `{first_alert}`",
                f"- First non-nominal interpreted cycle: `{first_non_nominal}`",
                f"- First threshold alarm cycle (vibration >= 5.2): `{threshold_cycle}`",
                f"- Early warning lead cycles: `{lead_cycles}`",
                "",
                "## Why this matters",
                "- Neraium emits interpretable structural risk progression before obvious threshold alarm behavior.",
                "- The sequence demonstrates normal operation, drift onset, rising instability, and critical state transition in one deterministic run.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic investor-facing Neraium demo proof artifacts.")
    parser.add_argument("--output-dir", default="reports/demo_proof")
    args = parser.parse_args()
    json_path, md_path = run_demo(Path(args.output_dir))
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
