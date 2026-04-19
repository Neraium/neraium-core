"""
Composite structural pressure / transition score from multiple scalar signals.

Missing values are tolerated; confidence reflects signal agreement.
"""

from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1e-12


def _normalize_scalar(x: float, lo: float, hi: float) -> float:
    span = hi - lo
    if span < EPS:
        return 0.5
    return float(np.clip((x - lo) / span, 0.0, 1.0))


def composite_structural_pressure(
    signals: dict[str, float],
    weights: dict[str, float] | None = None,
    *,
    missing_weight_policy: str = "renormalize",
) -> tuple[float, dict[str, Any]]:
    """
    Weighted normalized composite in [0, 1] plus diagnostics.

    Parameters
    ----------
    signals
        Map of signal name -> value (finite floats).
    weights
        Optional weights; defaults to equal weight over present keys.
    missing_weight_policy
        ``renormalize``: drop missing keys and scale weights to sum to 1.
        ``zero``: missing keys contribute 0.

    Returns
    -------
    score
        Composite score in [0, 1].
    meta
        Includes ``signals_used``, ``weights_effective``, ``agreement`` (pairwise variance penalty).
    """
    if not signals:
        return 0.5, {"signals_used": [], "weights_effective": {}, "agreement": 1.0, "reason": "no_signals"}

    w = dict(weights) if weights else {k: 1.0 for k in signals}
    present = {k: float(v) for k, v in signals.items() if k in w and np.isfinite(v)}
    if not present:
        return 0.5, {"signals_used": [], "weights_effective": {}, "agreement": 0.0, "reason": "no_finite_signals"}

    vals = np.asarray(list(present.values()), dtype=float)
    lo, hi = float(np.min(vals)), float(np.max(vals))
    norm_map = {k: _normalize_scalar(float(v), lo, hi) for k, v in present.items()}

    eff_w = {k: float(w[k]) for k in norm_map if k in w and float(w.get(k, 0.0)) > 0}
    if not eff_w:
        return 0.5, {"signals_used": list(present.keys()), "weights_effective": {}, "agreement": 0.0}

    if missing_weight_policy == "renormalize":
        s = sum(eff_w.values())
        eff_w = {k: v / s for k, v in eff_w.items()}
    else:
        s = sum(w.values())
        eff_w = {k: eff_w[k] / s for k in eff_w}

    score = sum(norm_map[k] * eff_w[k] for k in eff_w)
    score = float(np.clip(score, 0.0, 1.0))

    nvec = np.asarray([norm_map[k] for k in eff_w], dtype=float)
    agreement = float(1.0 - min(1.0, float(np.std(nvec)) * 2.0)) if len(nvec) > 1 else 1.0

    return score, {
        "signals_used": list(eff_w.keys()),
        "weights_effective": eff_w,
        "agreement": agreement,
        "normalization_span": {"min": lo, "max": hi},
    }
