"""Day 4: regime classification, confidence, interpretive gate, and action posture."""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Regime taxonomy (rules-based, deterministic) --------------------------------

REGIME_LABELS: frozenset[str] = frozenset(
    {
        "stable_trend",
        "fragile_rally",
        "risk_off_transition",
        "high_volatility",
        "unstable",
        "mean_reversion",
        "false_calm",
    }
)

# Action posture vocabulary
ACTION_POSTURES: frozenset[str] = frozenset(
    {
        "watch",
        "lean_long",
        "lean_short",
        "reduce_exposure",
        "avoid_risk",
        "wait",
    }
)

GATE_ACTIONS: frozenset[str] = frozenset(
    {"proceed", "no_action", "avoid_market", "wait"}
)


def _pct_rank(s: pd.Series) -> pd.Series:
    """Percent rank in [0, 1]; ``method='first'`` breaks ties so thresholds never all fail."""
    return s.rank(method="first", pct=True)


def classify_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign ``regime_label`` from structural + feature inputs.

    Rules are evaluated in priority order (first match wins):

    1. **risk_off_transition** — ``instability_score`` high *and* ``risk_off_proxy`` high
       (stress and cross-asset risk-off terms aligned).

    2. **fragile_rally** — ``instability_score`` rising vs 5d ago *and* breadth weakening
       (5d decline in ``breadth_pct_above_20dma``).

    3. **high_volatility** — realized vol high *and* correlation drift elevated
       (average of ``corr_drift_score`` and ``lag_drift_score``).

    4. **unstable** — ``coherence_score`` low *and* drift elevated (avg drift scores).

    5. **false_calm** — vol not elevated, drift scores not elevated, but drift metrics
       inflecting upward over 3d (sudden structural change while surface calm).

    6. **stable_trend** — ``instability_score`` low *and* ``corr_drift_score`` low.

    7. **mean_reversion** — none of the above (choppy / two-sided, no dominant narrative).

    Thresholds use percentile ranks on the input frame (``method='first'`` breaks ties).
    """
    out = df.copy()
    inst = out["instability_score"].astype(float)
    cd = out["corr_drift_score"].astype(float)
    ld = out["lag_drift_score"].astype(float)
    coh = out["coherence_score"].astype(float)
    drift_avg = (cd + ld) / 2.0

    risk_off = out.get("risk_off_proxy", pd.Series(0.0, index=out.index)).astype(float)
    breadth = out.get("breadth_pct_above_20dma", pd.Series(50.0, index=out.index)).astype(float)
    spy_vol = out.get("spy_vol_20d", pd.Series(np.nan, index=out.index)).astype(float)

    inst_rk = _pct_rank(inst)
    cd_rk = _pct_rank(cd)
    coh_rk = _pct_rank(coh)
    drift_rk = _pct_rank(drift_avg)
    risk_rk = _pct_rank(risk_off)

    inst_hi = inst_rk >= 0.66
    inst_lo = inst_rk <= 0.33
    cd_lo = cd_rk <= 0.33
    coh_lo = coh_rk <= 0.33
    drift_hi = drift_rk >= 0.66
    risk_hi = risk_rk >= 0.66

    inst_rising = inst.diff(5) > 0
    breadth_weak = breadth.diff(5) < 0

    if spy_vol.notna().any():
        sv = spy_vol.fillna(spy_vol.median())
        vol_rk = _pct_rank(sv)
        vol_hi = vol_rk >= 0.66
        vol_lo = vol_rk <= 0.33
    else:
        vol_hi = inst_hi
        vol_lo = inst_lo

    drift_lo = drift_rk <= 0.33
    sudden = (cd.diff(3) > 0) & (ld.diff(3) > 0)

    m_risk_off = inst_hi & risk_hi
    m_fragile = inst_rising.fillna(False) & breadth_weak.fillna(False)
    m_high_vol = vol_hi.fillna(False) & drift_hi
    m_unstable = coh_lo & drift_hi
    m_false_calm = vol_lo.fillna(False) & drift_lo.fillna(False) & sudden.fillna(False)
    m_stable = inst_lo & cd_lo

    label = np.select(
        [
            m_risk_off,
            m_fragile,
            m_high_vol,
            m_unstable,
            m_false_calm,
            m_stable,
        ],
        [
            "risk_off_transition",
            "fragile_rally",
            "high_volatility",
            "unstable",
            "false_calm",
            "stable_trend",
        ],
        default="mean_reversion",
    )
    out["regime_label"] = label
    return out


def compute_confidence_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ``confidence_score`` in [0, 1].

    Combines:

    - **coherence** (positive weight 0.4): markets moving together.
    - **1 - instability** (0.3): penalty when structural stress is high.
    - **signal_agreement** (0.3): inverse dispersion of normalized inputs
      (coherence, stability, breadth) — high when inputs agree.

    Clamped to [0, 1].
    """
    out = df.copy()
    inst = out["instability_score"].astype(float).clip(0, 1)
    coh = out["coherence_score"].astype(float).clip(0, 1)
    breadth = out.get("breadth_pct_above_20dma", pd.Series(50.0, index=out.index)).astype(float)
    breadth_norm = (breadth / 100.0).clip(0, 1)

    mat = pd.concat(
        [
            coh,
            (1.0 - inst),
            breadth_norm,
        ],
        axis=1,
    )
    std = mat.std(axis=1, ddof=0).fillna(0.0)
    signal_agreement = (1.0 - (std * 3.0).clip(0.0, 1.0)).clip(0.0, 1.0)

    conf = 0.4 * coh + 0.3 * (1.0 - inst) + 0.3 * signal_agreement
    out["confidence_score"] = conf.fillna(0.0).clip(0.0, 1.0)
    return out


def apply_interpretive_gate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpretive gate: suppress actions when evidence is weak or contradictory.

    Sets ``gate_action``:

    - **no_action** — ``confidence_score`` < 0.5 (insufficient conviction).
    - **avoid_market** — tail instability (>= 90th pct) *and* ``coherence_score`` in
      bottom third (chaotic, incoherent).
    - **wait** — cross-signal conflict: ``risk_off_proxy`` and ``spy_ret_1d`` push in
      opposite directions with both sufficiently large (mixed narrative).
    - **proceed** — otherwise.

    Downstream posture uses this to abstain or override.
    """
    out = df.copy()
    conf = out["confidence_score"].astype(float)
    inst = out["instability_score"].astype(float)
    coh = out["coherence_score"].astype(float)
    risk_off = out.get("risk_off_proxy", pd.Series(0.0, index=out.index)).astype(float)
    spy_r = out.get("spy_ret_1d", pd.Series(0.0, index=out.index)).astype(float)

    inst_tail = _pct_rank(inst) >= 0.90
    coh_low = _pct_rank(coh) <= 0.33

    ro_mag = _pct_rank(risk_off.abs()) >= 0.5
    spy_mag = _pct_rank(spy_r.abs()) >= 0.5
    conflict = (np.sign(risk_off) != np.sign(spy_r)) & ro_mag & spy_mag

    n = len(out)
    gate = np.full(n, "proceed", dtype=object)
    gate[conf.to_numpy() < 0.5] = "no_action"
    m_avoid = (gate == "proceed") & inst_tail.to_numpy() & coh_low.to_numpy()
    gate[m_avoid] = "avoid_market"
    m_wait = (gate == "proceed") & conflict.to_numpy()
    gate[m_wait] = "wait"
    out["gate_action"] = gate.astype(str)
    return out


def generate_action_posture(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map regime, confidence, and gate into ``action_posture``.

    Gate overrides: ``no_action`` / ``wait`` / ``avoid_market`` map to ``wait`` or
    ``avoid_risk``. If ``proceed``, regime drives stance:

    - **stable_trend** + high confidence → **lean_long**
    - **stable_trend** → **watch**
    - **fragile_rally** → **reduce_exposure**
    - **risk_off_transition** → **avoid_risk**
    - **high_volatility** → **reduce_exposure**
    - **unstable** → **wait**
    - **false_calm** → **watch**
    - **mean_reversion** → **watch**

    ``lean_short`` is reserved for future bearish rules; not emitted in default rules.
    """
    out = df.copy()
    regime = out["regime_label"].astype(str)
    conf = out["confidence_score"].astype(float)
    gate = out["gate_action"].astype(str)

    r = regime.to_numpy()
    high_conf = (conf >= 0.65).to_numpy()

    base = np.select(
        [
            (r == "stable_trend") & high_conf,
            (r == "stable_trend"),
            (r == "fragile_rally"),
            (r == "risk_off_transition"),
            (r == "high_volatility"),
            (r == "unstable"),
            (r == "false_calm"),
            (r == "mean_reversion"),
        ],
        [
            "lean_long",
            "watch",
            "reduce_exposure",
            "avoid_risk",
            "reduce_exposure",
            "wait",
            "watch",
            "watch",
        ],
        default="watch",
    )

    posture = np.array(base, dtype=object)
    g = gate.to_numpy()
    posture = np.where(g == "no_action", "wait", posture)
    posture = np.where(g == "wait", "wait", posture)
    posture = np.where(g == "avoid_market", "avoid_risk", posture)
    out["action_posture"] = posture.astype(str)
    return out
