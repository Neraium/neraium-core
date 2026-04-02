"""Day 4 signal generation: structural snapshot → regime → confidence → gate → posture."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DATE_COLUMN
from neraium.regime import (
    apply_interpretive_gate,
    classify_regime,
    compute_confidence_score,
    generate_action_posture,
)
from neraium.structural import build_structural_snapshot


def _fmt(x: float, nd: int = 2) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{float(x):.{nd}f}"


def _build_explanation(row: pd.Series) -> str:
    """Single-row human-readable rationale (deterministic string from row state)."""
    regime = str(row.get("regime_label", ""))
    conf = row.get("confidence_score", float("nan"))
    inst = row.get("instability_score", float("nan"))
    coh = row.get("coherence_score", float("nan"))
    gate = str(row.get("gate_action", "proceed"))
    posture = str(row.get("action_posture", "watch"))

    parts: list[str] = [
        f"Regime={regime}",
        f"confidence={_fmt(conf)}",
        f"instability={_fmt(inst)}",
        f"coherence={_fmt(coh)}",
        f"gate={gate}",
        f"posture={posture}",
    ]

    if regime == "fragile_rally":
        parts.append(
            "Instability rising with weaker breadth suggests a fragile rally; size risk carefully."
        )
    elif regime == "risk_off_transition":
        parts.append("Elevated stress and risk-off cross-asset tilt favour defensive positioning.")
    elif regime == "high_volatility":
        parts.append("High realized volatility with elevated correlation drift implies choppy conditions.")
    elif regime == "unstable":
        parts.append("Low coherence with elevated drift indicates an unstable, hard-to-trust structure.")
    elif regime == "false_calm":
        parts.append("Surface calm with emerging structural drift warrants vigilance.")
    elif regime == "stable_trend":
        parts.append("Low instability and contained correlation drift support a calmer trend read.")
    else:
        parts.append("Mixed or two-sided structure consistent with mean-reverting behaviour.")

    return "; ".join(parts)


def generate_signals(structural_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Day 4 pipeline on a structural snapshot (Day 3 output).

    Returns columns:
    ``timestamp``, ``regime_label``, ``confidence_score``, ``action_posture``,
    ``instability_score``, ``coherence_score``, ``explanation``.
    """
    df = structural_df.copy()
    df = classify_regime(df)
    df = compute_confidence_score(df)
    df = apply_interpretive_gate(df)
    df = generate_action_posture(df)
    df["explanation"] = df.apply(_build_explanation, axis=1)

    out_cols = [
        DATE_COLUMN,
        "regime_label",
        "confidence_score",
        "action_posture",
        "instability_score",
        "coherence_score",
        "explanation",
    ]
    out = df[out_cols].sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    return out


def run_full_signal_pipeline(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Build structural snapshot from features, then generate signals."""
    snap = build_structural_snapshot(feature_df)
    return generate_signals(snap)


def save_signals_csv(signals: pd.DataFrame, path: Path | None = None) -> Path:
    """Write signals to CSV under ``output/signals.csv`` by default."""
    root = Path(__file__).resolve().parent.parent
    out = path or (root / "output" / "signals.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(out, index=False)
    return out
