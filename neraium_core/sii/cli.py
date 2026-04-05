from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neraium_core.sii.orchestration import run_structural_pipeline
from neraium_core.sii.reporting import format_structural_report_text, generate_structural_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m neraium_core.sii.cli",
        description="Run the SII structural pipeline on a local dataset file.",
    )
    parser.add_argument("--input", help="Input dataset path (.json primary, .csv optional).")
    parser.add_argument("--input-dir", help="Input directory containing JSON/CSV datasets.")
    parser.add_argument("--batch", action="store_true", help="Run all files from --input-dir in deterministic order.")
    parser.add_argument("--watch", help="Watch a directory and process new JSON/CSV files as they appear.")
    parser.add_argument(
        "--watch-iterations",
        type=int,
        default=0,
        help="Optional number of polling iterations for watch mode (0 = run indefinitely).",
    )
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Watch polling interval in seconds.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file JSON output and show only summary/report.")
    parser.add_argument("--report", action="store_true", help="Print per-file report and aggregate structural report.")
    parser.add_argument("--aggregate-output", help="Optional output file path for aggregate run JSON.")
    parser.add_argument("--output", help="Optional output file path for structured JSON.")
    parser.add_argument("--report-file", help="Optional output file path for human-readable text report.")
    return parser


@dataclass
class ExecutionState:
    processed_files: set[str] = field(default_factory=set)
    discovered_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    last_run_timestamp: str | None = None
    total_runs: int = 0
    risk_flag_count: int = 0
    high_risk_count: int = 0
    signal_counts: Counter[str] = field(default_factory=Counter)
    triggered_signal_lead_recall_total: float = 0.0
    triggered_signal_count: int = 0

    def update(self, run_output: dict[str, Any]) -> None:
        self.total_runs += 1
        self.last_run_timestamp = datetime.now(timezone.utc).isoformat()

        decision = run_output.get("decision_output") if isinstance(run_output.get("decision_output"), dict) else {}
        ranking = run_output.get("signal_ranking") if isinstance(run_output.get("signal_ranking"), dict) else {}
        ranking_list = ranking.get("ranking") if isinstance(ranking.get("ranking"), list) else []
        lead_recall_by_signal: dict[str, float] = {}
        for item in ranking_list:
            if not isinstance(item, dict):
                continue
            signal_name = str(item.get("signal") or "")
            predictive = item.get("predictive_metrics") if isinstance(item.get("predictive_metrics"), dict) else {}
            if signal_name:
                lead_recall_by_signal[signal_name] = float(predictive.get("lead_recall") or 0.0)

        risk_flag = bool(decision.get("risk_flag"))
        if risk_flag:
            self.risk_flag_count += 1
        risk_level = str(decision.get("risk_level") or "").lower()
        if risk_level == "high":
            self.high_risk_count += 1

        contributing = decision.get("contributing_signals")
        if isinstance(contributing, list):
            for entry in contributing:
                signal_name = str((entry if isinstance(entry, dict) else {}).get("signal") or "")
                if signal_name:
                    self.signal_counts[signal_name] += 1
                    self.triggered_signal_lead_recall_total += lead_recall_by_signal.get(signal_name, 0.0)
                    self.triggered_signal_count += 1

    def aggregate_summary(self) -> dict[str, Any]:
        most_frequent = ""
        if self.signal_counts:
            most_frequent = sorted(self.signal_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        avg_lead_recall = (
            self.triggered_signal_lead_recall_total / self.triggered_signal_count if self.triggered_signal_count > 0 else 0.0
        )
        return {
            "total_discovered_files": self.discovered_files,
            "total_runs": self.total_runs,
            "successful_runs": self.total_runs,
            "failed_files": self.failed_files,
            "skipped_files": self.skipped_files,
            "risk_flag_true_count": self.risk_flag_count,
            "high_risk_count": self.high_risk_count,
            "most_frequent_contributing_signals": self.signal_counts.most_common(),
            "most_frequent_signal": most_frequent,
            "average_triggered_signal_lead_recall": round(avg_lead_recall, 6),
            "last_run_timestamp": self.last_run_timestamp,
        }


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in input file: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError("JSON input must be a list of record objects")

    records: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"JSON record at index {index} must be an object")
        records.append(row)
    return records


def _load_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV input must include a header row")

        metadata_keys = {"timestamp", "site_id", "system_id", "asset_id", "node_id"}
        records: list[dict[str, Any]] = []
        for index, row in enumerate(reader):
            sensor_values: dict[str, float | None] = {}
            for key, raw_value in row.items():
                if key is None or key in metadata_keys:
                    continue
                text = (raw_value or "").strip()
                if text == "":
                    sensor_values[key] = None
                    continue
                try:
                    sensor_values[key] = float(text)
                except ValueError as exc:
                    raise ValueError(f"CSV value at row {index + 2}, column {key!r} must be numeric") from exc

            records.append(
                {
                    "timestamp": row.get("timestamp"),
                    "site_id": row.get("site_id"),
                    "system_id": row.get("system_id"),
                    "asset_id": row.get("asset_id"),
                    "node_id": row.get("node_id"),
                    "sensor_values": sensor_values,
                }
            )

    return records


def _load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json_records(path)
    if suffix == ".csv":
        return _load_csv_records(path)
    raise ValueError(f"Unsupported input file type: {path.suffix or '<none>'}. Use .json or .csv")


def _iter_input_files(input_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in input_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".json", ".csv"}:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.name)


def _build_run_result(path: Path, output: dict[str, Any]) -> dict[str, Any]:
    return {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "source_file": path.name,
        "source_path": str(path),
        "status": "success",
        "result": output,
    }


def _build_failed_result(path: Path, error: Exception) -> dict[str, Any]:
    return {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "source_file": path.name,
        "source_path": str(path),
        "status": "failed",
        "error": {
            "type": error.__class__.__name__,
            "message": str(error),
        },
    }


def _build_skipped_result(path: Path, reason: str) -> dict[str, Any]:
    return {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "source_file": path.name,
        "source_path": str(path),
        "status": "skipped",
        "skip_reason": reason,
    }


def _print_per_file_report(run_result: dict[str, Any]) -> None:
    status = str(run_result.get("status") or "unknown")
    if status != "success":
        error = run_result.get("error") if isinstance(run_result.get("error"), dict) else {}
        skip_reason = run_result.get("skip_reason")
        print(
            (
                f"[REPORT] file={run_result.get('source_file')} "
                f"status={status} "
                f"detail={error.get('message') or skip_reason}"
            )
        )
        return
    result = run_result.get("result") if isinstance(run_result.get("result"), dict) else {}
    decision = result.get("decision_output") if isinstance(result.get("decision_output"), dict) else {}
    print(
        (
            f"[REPORT] file={run_result.get('source_file')} "
            f"risk_flag={bool(decision.get('risk_flag'))} "
            f"risk_level={decision.get('risk_level')}"
        )
    )


def _print_aggregate_report(aggregate_summary: dict[str, Any]) -> None:
    print("==== Aggregate Structural Summary ====")
    print(f"Discovered Files: {aggregate_summary['total_discovered_files']}")
    print(f"Total Runs: {aggregate_summary['total_runs']}")
    print(f"Failed Files: {aggregate_summary['failed_files']}")
    print(f"Skipped Files: {aggregate_summary['skipped_files']}")
    print(f"Risk Events: {aggregate_summary['risk_flag_true_count']}")
    print(f"High Risk: {aggregate_summary['high_risk_count']}")
    print(f"Most Frequent Signal: {aggregate_summary['most_frequent_signal'] or 'n/a'}")


def _process_file(
    path: Path,
    state: ExecutionState,
    *,
    quiet: bool,
    report: bool,
    emit_wrapped_result: bool = True,
) -> dict[str, Any]:
    records = _load_records(path)
    output = run_structural_pipeline(records)
    run_result = _build_run_result(path, output)
    state.processed_files.add(str(path.resolve()))
    state.update(output)
    return run_result


def _is_file_size_stable(path: Path) -> bool:
    first = path.stat().st_size
    time.sleep(0.05)
    second = path.stat().st_size
    return first == second


def _process_file_safe(
    path: Path,
    state: ExecutionState,
    *,
    quiet: bool,
    report: bool,
    emit_wrapped_result: bool = True,
) -> dict[str, Any]:
    state.discovered_files += 1
    resolved = str(path.resolve())
    if resolved in state.processed_files:
        state.skipped_files += 1
        run_result = _build_skipped_result(path, "already_processed")
    elif not _is_file_size_stable(path):
        state.skipped_files += 1
        run_result = _build_skipped_result(path, "partial_write_unstable_size")
    else:
        try:
            run_result = _process_file(
                path,
                state,
                quiet=quiet,
                report=report,
                emit_wrapped_result=emit_wrapped_result,
            )
        except Exception as exc:
            state.failed_files += 1
            run_result = _build_failed_result(path, exc)

    if not quiet:
        stdout_payload: dict[str, Any] = run_result
        if run_result.get("status") == "success" and not emit_wrapped_result:
            stdout_payload = run_result.get("result") if isinstance(run_result.get("result"), dict) else run_result
        print(json.dumps(stdout_payload, indent=2, sort_keys=True))
    if report:
        _print_per_file_report(run_result)
    return run_result


def _write_json(path: str, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: str, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.input and not args.input_dir and not args.watch:
        print("Provide one of --input, --input-dir, or --watch.", file=sys.stderr)
        return 2
    if args.input and (args.input_dir or args.watch):
        print("--input cannot be combined with --input-dir or --watch.", file=sys.stderr)
        return 2

    state = ExecutionState()
    run_results: list[dict[str, Any]] = []
    if args.watch:
        watch_dir = Path(args.watch)
        if not watch_dir.exists() or not watch_dir.is_dir():
            print(f"Watch directory not found: {watch_dir}", file=sys.stderr)
            return 2
        iteration = 0
        while True:
            try:
                files = _iter_input_files(watch_dir)
            except Exception as exc:
                print(f"Watch iteration error: {exc}", file=sys.stderr)
                files = []

            for path in files:
                run_results.append(_process_file_safe(path, state, quiet=args.quiet, report=args.report))

            if run_results and not args.quiet:
                print(json.dumps({"aggregate_summary": state.aggregate_summary()}, indent=2, sort_keys=True))

            iteration += 1
            if args.watch_iterations > 0 and iteration >= args.watch_iterations:
                break
            time.sleep(max(0.1, float(args.poll_interval)))
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            print(f"Input directory not found: {input_dir}", file=sys.stderr)
            return 2
        for path in _iter_input_files(input_dir):
            run_results.append(_process_file_safe(path, state, quiet=args.quiet, report=args.report))
    else:
        input_path = Path(args.input)
        if not input_path.exists() or not input_path.is_file():
            print(f"Input file not found: {input_path}", file=sys.stderr)
            return 2
        run_results.append(
            _process_file_safe(
                input_path,
                state,
                quiet=args.quiet,
                report=args.report,
                emit_wrapped_result=False,
            )
        )

    aggregate_summary = state.aggregate_summary()
    aggregate_payload = {"runs": run_results, "aggregate_summary": aggregate_summary}
    latest_result: dict[str, Any] = {}
    for item in reversed(run_results):
        candidate = item.get("result") if isinstance(item.get("result"), dict) else None
        if candidate is not None:
            latest_result = candidate
            break
    report_text = format_structural_report_text(generate_structural_report(latest_result)) if latest_result else ""

    if args.output:
        _write_json(args.output, latest_result)
    if args.aggregate_output:
        _write_json(args.aggregate_output, aggregate_payload)
    if args.report_file:
        _write_text(args.report_file, report_text)

    if args.batch or args.watch or args.quiet:
        print(json.dumps({"aggregate_summary": aggregate_summary}, indent=2, sort_keys=True))
    if args.report:
        if report_text:
            print(report_text)
        _print_aggregate_report(aggregate_summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
