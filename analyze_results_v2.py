#!/usr/bin/env python3
"""
Print a human-readable analysis of pilot_results_v2.json (stdout only; no files written).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def _load_document(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at root, got {type(data).__name__}")
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError('Expected top-level "records" array')
    return data


def _first_step_for_state(records: list[dict[str, Any]], state: str) -> int | None:
    for r in records:
        if str(r.get("state")) == state:
            return int(r["timestep"])
    return None


def _scores(records: list[dict[str, Any]]) -> list[float]:
    out: list[float] = []
    for r in records:
        s = r.get("score")
        if s is None:
            continue
        try:
            out.append(float(s))
        except (TypeError, ValueError):
            pass
    return out


def _missing_data_count(records: list[dict[str, Any]]) -> int:
    return sum(1 for r in records if r.get("missing_data") is True)


def _duplicate_count(records: list[dict[str, Any]]) -> int:
    return sum(1 for r in records if r.get("frame_type") == "duplicate")


def _flatline_detected(
    records: list[dict[str, Any]], *, ref: float = 9.5, tol: float = 0.05
) -> tuple[bool, int, int]:
    """
    Heuristic: timesteps >= 75 with s4 pinned near `ref` (matches pilot scenario s4 flatline).
    Returns (detected, matching_steps, post_threshold_steps).
    """
    post = [
        r for r in records if int(r["timestep"]) >= 75 and r.get("frame_type") != "duplicate"
    ]
    if not post:
        return False, 0, 0

    matches = 0
    for r in post:
        sig = r.get("signals")
        if not isinstance(sig, dict):
            continue
        s4 = sig.get("s4")
        if s4 is None:
            continue
        try:
            if abs(float(s4) - ref) <= tol:
                matches += 1
        except (TypeError, ValueError):
            pass

    n = len(post)
    detected = matches >= max(3, (n + 1) // 2)
    return detected, matches, n


def _fmt_num(x: float | None, places: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{places}f}"


def _print_report(path: Path) -> None:
    doc = _load_document(path)
    records = [r for r in doc["records"] if isinstance(r, dict)]

    first_watch = _first_step_for_state(records, "WATCH")
    first_alert = _first_step_for_state(records, "ALERT")

    states = [str(r.get("state", "?")) for r in records]
    interps = [str(r.get("interpreted_state", "?")) for r in records]
    state_counts = Counter(states)
    interp_counts = Counter(interps)

    scores = _scores(records)
    max_score = max(scores) if scores else None
    mean_all = statistics.mean(scores) if scores else None

    scores_ge_60: list[float] = []
    for r in records:
        if int(r["timestep"]) < 60:
            continue
        s = r.get("score")
        if s is None:
            continue
        try:
            scores_ge_60.append(float(s))
        except (TypeError, ValueError):
            pass
    mean_60 = statistics.mean(scores_ge_60) if scores_ge_60 else None

    n_missing = _missing_data_count(records)
    n_dup = _duplicate_count(records)
    flat_ok, flat_hits, flat_n = _flatline_detected(records)

    print("=" * 60)
    print(f"Pilot results analysis: {path.resolve()}")
    print("=" * 60)
    print()
    print("Summary")
    print("-" * 40)
    print(f"  First WATCH step:     {first_watch if first_watch is not None else 'n/a'}")
    print(f"  First ALERT step:     {first_alert if first_alert is not None else 'n/a'}")
    print()
    print("  Count by state (pilot / score-threshold):")
    for k in ("STABLE", "WATCH", "ALERT"):
        if k in state_counts:
            print(f"    {k}: {state_counts[k]}")
    for k, v in sorted(state_counts.items()):
        if k not in ("STABLE", "WATCH", "ALERT"):
            print(f"    {k}: {v}")
    print()
    print("  Count by interpreted_state (smoothed):")
    for k, v in sorted(interp_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {k}: {v}")
    print()
    print(f"  Max score:                 {_fmt_num(max_score)}")
    print(f"  Mean score (overall):      {_fmt_num(mean_all)}")
    print(f"  Mean score (timestep ≥60): {_fmt_num(mean_60)}")
    print()
    print(f"  Missing-data steps:        {n_missing}")
    print(f"  Duplicate frames:          {n_dup}")
    print(
        f"  Flatline behavior:         "
        f"{'yes' if flat_ok else 'no'} "
        f"({flat_hits}/{flat_n} steps at t≥75 with s4≈9.5, excl. duplicates)"
    )
    print()

    # --- Interpretation ---
    n = len(records)
    print("Interpretation")
    print("-" * 40)
    lines: list[str] = []

    if n == 0:
        lines.append("No records were present in the file.")
    else:
        lines.append(
            f"The run contains {n} ingest rows (logical timesteps plus any duplicate ingests)."
        )
        if first_watch is not None:
            lines.append(
                f"The pilot crossed into WATCH (score in [1.25, 2.0)) starting at timestep {first_watch}."
            )
        if first_alert is not None:
            lines.append(
                f"ALERT (score ≥ 2.0) first appears at timestep {first_alert}, "
                "indicating the strongest instability band in this scenario."
            )
        if max_score is not None and mean_all is not None:
            lines.append(
                f"Scores range up to {_fmt_num(max_score)} with an overall average of {_fmt_num(mean_all)}."
            )
        if mean_60 is not None:
            lines.append(
                f"After timestep 60 (stronger disruption phase), the average score is {_fmt_num(mean_60)}, "
                "typically higher than the early baseline period."
            )
        if n_missing:
            lines.append(
                f"{n_missing} step(s) are flagged with missing sensor data, which usually widens uncertainty "
                "and can depress or distort scores depending on the engine."
            )
        if n_dup:
            lines.append(
                f"{n_dup} duplicate frame(s) were replayed; pilot score/state for those rows should match "
                "the first ingest at the same timestep."
            )
        if flat_ok:
            lines.append(
                "Sensor s4 shows a sustained flat value in the late window (t≥75), consistent with an "
                "injected flatline / stuck-sensor behavior in the pilot scenario."
            )
        else:
            lines.append(
                "No clear s4 flatline signature was detected in the t≥75 window with the default heuristic "
                "(or there were too few post-75 rows)."
            )

        top_interp = interp_counts.most_common(1)
        if top_interp:
            name, cnt = top_interp[0]
            lines.append(
                f"The dominant smoothed interpretation is {name} ({cnt} rows), reflecting hysteresis on "
                "the engine’s raw interpreted_state."
            )

    for line in lines:
        print(f"  {line}")
    print()
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pilot_results_v2.json (print only).")
    parser.add_argument(
        "path",
        nargs="?",
        default="pilot_results_v2.json",
        type=Path,
        help="Path to pilot results JSON (default: pilot_results_v2.json)",
    )
    args = parser.parse_args()
    path: Path = args.path
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")
    _print_report(path)


if __name__ == "__main__":
    main()
