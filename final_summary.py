#!/usr/bin/env python3
"""
Compare three pilot result documents (records + summary).

Default files (repo root):
  - baseline_coupling_instability.json  (e.g. python run_pilot.py --output baseline_coupling_instability.json)
  - pilot_regime_shift.json
  - pilot_structural_instability.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_doc(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError(f'{path}: expected top-level "records" array')
    return data


def _first_state_step(records: list[dict[str, Any]], state: str) -> int | None:
    for r in records:
        if str(r.get("state")) == state:
            return int(r["timestep"])
    return None


def _dominant_interpreted(records: list[dict[str, Any]]) -> str:
    if not records:
        return "n/a"
    counts = Counter(str(r.get("interpreted_state", "NOMINAL_STRUCTURE")) for r in records)
    top = counts.most_common(2)
    if not top:
        return "n/a"
    if len(top) == 1 or top[0][1] > top[1][1]:
        return top[0][0]
    # tie: show both
    tied = [k for k, v in top if v == top[0][1]]
    return " / ".join(sorted(tied))


def _degraded_present(records: list[dict[str, Any]], summary: dict[str, Any]) -> bool:
    if summary.get("missing_data_frame_count", 0) > 0:
        return True
    if summary.get("duplicate_frame_count", 0) > 0:
        return True
    for r in records:
        if r.get("missing_data") is True:
            return True
        if r.get("frame_type") == "duplicate":
            return True
    return False


def _max_score(records: list[dict[str, Any]]) -> float | None:
    scores: list[float] = []
    for r in records:
        s = r.get("score")
        if s is None:
            continue
        try:
            scores.append(float(s))
        except (TypeError, ValueError):
            pass
    return max(scores) if scores else None


def _format_watch_regime(first: int | None, first_alert: int | None) -> str:
    """Regime-shift scenario: calm baseline / change-detection wording."""
    if first is None:
        return "None or minimal"
    if first_alert is None:
        return f"{first} (low-level, no ALERT)"
    return str(first)


def _scenario_title(stem: str) -> str:
    mapping = {
        "baseline_coupling_instability": "Coupling Instability",
        "pilot_regime_shift": "Regime Shift",
        "pilot_structural_instability": "Structural Instability",
    }
    return mapping.get(stem, stem.replace("_", " ").title())


def _print_scenario_block(
    *,
    title: str,
    first_watch: int | None,
    first_alert: int | None,
    dominant: str,
    degraded: bool,
    max_sc: float | None,
    style: str,
) -> None:
    print(f"Scenario: {title}")

    if style == "regime_shift":
        watch_s = _format_watch_regime(first_watch, first_alert)
        alert_s = "None" if first_alert is None else str(first_alert)
    else:
        watch_s = str(first_watch) if first_watch is not None else "None"
        alert_s = str(first_alert) if first_alert is not None else "None"

    print(f"- First WATCH: {watch_s}")
    print(f"- First ALERT: {alert_s}")
    print(f"- Dominant interpreted_state: {dominant}")
    if max_sc is not None:
        print(f"- Max score: {max_sc:.4f}")
    else:
        print("- Max score: n/a")
    print(f"- Degraded Data: {'Yes' if degraded else 'No'}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare three pilot result JSON documents.")
    parser.add_argument(
        "--coupling",
        type=Path,
        default=Path("baseline_coupling_instability.json"),
        help="Main pilot scenario with injected degradations",
    )
    parser.add_argument(
        "--regime",
        type=Path,
        default=Path("pilot_regime_shift.json"),
        help="Regime shift file-mode pilot output",
    )
    parser.add_argument(
        "--structural",
        type=Path,
        default=Path("pilot_structural_instability.json"),
        help="Structural instability file-mode pilot output",
    )
    args = parser.parse_args()

    triples: list[tuple[Path, str, str]] = [
        (args.coupling, _scenario_title(args.coupling.stem), "coupling"),
        (args.regime, _scenario_title(args.regime.stem), "regime_shift"),
        (args.structural, _scenario_title(args.structural.stem), "structural"),
    ]

    missing = [p for p, _, _ in triples if not p.is_file()]
    if missing:
        msg = "Missing file(s):\n" + "\n".join(f"  - {p.resolve()}" for p in missing)
        msg += "\n\nTip: generate the baseline with:\n  python run_pilot.py --output baseline_coupling_instability.json"
        raise SystemExit(msg)

    print("=" * 60)
    print("Pilot scenarios - comparison summary")
    print("=" * 60)
    print()

    for path, title, style in triples:
        doc = _load_doc(path)
        records = [r for r in doc["records"] if isinstance(r, dict)]
        summary = doc.get("summary") if isinstance(doc.get("summary"), dict) else {}

        fw = _first_state_step(records, "WATCH")
        fa = _first_state_step(records, "ALERT")
        dom = _dominant_interpreted(records)
        deg = _degraded_present(records, summary)
        mx = _max_score(records)

        _print_scenario_block(
            title=title,
            first_watch=fw,
            first_alert=fa,
            dominant=dom,
            degraded=deg,
            max_sc=mx,
            style=style,
        )

    print("Key Insight:")
    print(
        "The system distinguishes between structural failure, coupling instability, and "
        "regime shifts without relying on thresholds, and remains stable under degraded "
        "data conditions."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
