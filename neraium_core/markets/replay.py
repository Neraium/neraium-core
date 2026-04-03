"""Replay and paper-trading style validation for Neraium Markets."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from neraium_core.markets.data.alignment import align_to_shared_clock
from neraium_core.markets.data.market_data_loader import MarketDataLoader
from neraium_core.markets.data.validation import validate_market_frame
from neraium_core.markets.evidence.evidence_log import EvidenceLog
from neraium_core.markets.signals.emission_control import EmissionControlConfig, SignalEmissionController
from neraium_core.markets.signals.signal_generator import generate_signal_for_asset, to_evidence
from neraium_core.markets.state.state_vector import CORE, build_state_vector


def run_signal_replay(
    data_dir: Path,
    evidence_log: EvidenceLog,
    *,
    timeframe: str = "15m",
    csv_output_path: Path | None = None,
    emission_controller: SignalEmissionController | None = None,
    symbols: list[str] | None = None,
) -> list[dict]:
    loader = MarketDataLoader(data_dir=data_dir)
    prices = align_to_shared_clock(loader.load(timeframe))
    minimum_history = max(30, min(80, len(prices.index)))
    if len(prices.index) < minimum_history:
        raise ValueError("Replay requires enough aligned rows for baseline/recent context")

    controller = emission_controller or SignalEmissionController(EmissionControlConfig())
    controller.config.warmup_bars = min(controller.config.warmup_bars, max(20, len(prices.index) // 2))
    emitted_rows: list[dict] = []
    target_assets = [s.upper() for s in (symbols or CORE)]

    for frame_index in range(minimum_history, len(prices.index) + 1):
        frame_prices = prices.iloc[:frame_index]
        validation = validate_market_frame(frame_prices)
        state = build_state_vector(frame_prices)
        timestamp = frame_prices.index[-1]
        for asset in target_assets:
            try:
                signal = generate_signal_for_asset(
                    asset=asset,
                    timeframe=timeframe,
                    state=state,
                    data_quality=float(validation["completeness"]),
                )
            except ValueError:
                continue
            if signal is None or signal.trader_output is None or signal.decision_validity is None:
                continue

            stable_permission = controller.stabilize_permission(
                f"{asset}:{timeframe}",
                signal.trader_output.action_permission,
                signal.trader_output.validity_score,
            )
            signal.trader_output.action_permission = stable_permission
            signal.trader_output.trading_signal = f"{stable_permission}:{signal.trader_output.best_action}"
            signal.decision_validity.action_permission = stable_permission
            signal.timestamp = pd.Timestamp(timestamp).to_pydatetime()
            signal.trader_output.timestamp = pd.Timestamp(timestamp).to_pydatetime()

            decision = controller.evaluate(
                f"{asset}:{timeframe}",
                frame_index=frame_index,
                sample_count=len(frame_prices),
                confidence=signal.trader_output.confidence,
                action_permission=signal.trader_output.action_permission,
                best_action=signal.trader_output.best_action,
                validity_score=signal.trader_output.validity_score,
                market_state=signal.trader_output.market_state,
                transition_state=signal.trader_output.transition_state,
                abstain=signal.decision_validity.abstain,
            )
            signal.trader_output.cooldown_status = decision.cooldown_status
            if not decision.emit:
                continue

            controller.mark_emitted(
                f"{asset}:{timeframe}",
                frame_index=frame_index,
                action_permission=signal.trader_output.action_permission,
                best_action=signal.trader_output.best_action,
                validity_score=signal.trader_output.validity_score,
                market_state=signal.trader_output.market_state,
                transition_state=signal.trader_output.transition_state,
            )
            evidence_log.append(
                to_evidence(
                    signal,
                    {
                        "validation": validation,
                        "timeframe": timeframe,
                        "replay_timestamp": timestamp.isoformat(),
                    },
                )
            )
            emitted_rows.append(signal.trader_output.model_dump(mode="json"))

    if csv_output_path is not None:
        csv_output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "timestamp",
            "ticker",
            "market_state",
            "transition_state",
            "action_permission",
            "best_action",
            "trading_signal",
            "confidence",
            "fragility",
            "validity_score",
            "structural_drift_score",
            "latest_instability",
            "one_line_reason",
            "size_guidance",
            "cooldown_status",
        ]
        with csv_output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in emitted_rows:
                writer.writerow({key: row.get(key) for key in fieldnames})

    return emitted_rows
