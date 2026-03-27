"""
Platform-agnostic structural transition detection from time-ordered replay data.

Separates signal generation (upstream) from event detection (this module).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from neraium_core.detection.transition_config import TransitionDetectionConfig

EPS = 1e-12


def _as_float_series(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    return out.interpolate(limit_direction="both").ffill().bfill()


def _normalize_01(raw: pd.Series) -> pd.Series:
    pmin = float(raw.min())
    pmax = float(raw.max())
    span = pmax - pmin
    if span < EPS:
        return pd.Series(0.5, index=raw.index, dtype=float)
    return ((raw - pmin) / span).clip(0.0, 1.0)


def _select_signal_column(df: pd.DataFrame, config: TransitionDetectionConfig) -> tuple[str | None, list[str]]:
    tried: list[str] = []
    if config.primary_signal and config.primary_signal in df.columns:
        return config.primary_signal, tried
    for col in config.signal_priority:
        tried.append(col)
        if col in df.columns:
            return col, tried
    return None, tried


def _build_composite_signal(df: pd.DataFrame, config: TransitionDetectionConfig) -> tuple[pd.Series | None, str]:
    weights = config.composite_weights or {}
    if not weights:
        weights = {c: 1.0 for c in config.signal_priority if c in df.columns}
    if not weights:
        return None, "no_composite_columns"
    total_w = 0.0
    acc = None
    parts: list[str] = []
    for col, w in weights.items():
        if col not in df.columns or w <= 0:
            continue
        raw = _as_float_series(df[col])
        nrm = _normalize_01(raw)
        acc = nrm * float(w) if acc is None else acc + nrm * float(w)
        total_w += float(w)
        parts.append(col)
    if acc is None or total_w < EPS:
        return None, "composite_empty"
    comp = (acc / total_w).clip(0.0, 1.0)
    label = "composite[" + "+".join(parts) + "]"
    return comp, label


def _apply_warmup_mask(
    df: pd.DataFrame,
    config: TransitionDetectionConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return dataframe after warmup exclusion and metadata."""
    meta: dict[str, Any] = {"rows_in": len(df)}
    idx = pd.to_numeric(df[config.index_column], errors="coerce")
    df = df.copy()

    if config.warmup_fraction is not None and 0 < config.warmup_fraction < 1:
        k = int(len(df) * float(config.warmup_fraction))
        k = max(0, min(k, len(df) - 1))
        df = df.iloc[k:].copy()
        meta["warmup_mode"] = "fraction"
        meta["warmup_rows_dropped"] = k
    else:
        wc = float(config.warmup_cycles)
        if idx.notna().any() and np.nanmax(idx.values) >= wc:
            df = df.loc[idx >= wc].copy()
            meta["warmup_mode"] = "index_gte"
            meta["warmup_cycles"] = wc
        else:
            # Non-numeric or short series: drop first N rows by position
            n_drop = min(int(max(0, config.warmup_cycles)), max(0, len(df) - 1))
            if n_drop > 0:
                df = df.iloc[n_drop:].copy()
            meta["warmup_mode"] = "head_drop"
            meta["warmup_rows_dropped"] = int(n_drop)

    df = df.sort_values(config.index_column)
    meta["rows_after_warmup"] = len(df)
    return df, meta


def detect_transition_from_signal(
    signal: pd.Series | np.ndarray,
    index: pd.Series | np.ndarray | None = None,
    config: TransitionDetectionConfig | None = None,
) -> dict[str, Any]:
    """
    Run detection on a single 1D signal with optional parallel index (positions).

    ``index`` should align with ``signal`` (same length). If omitted, uses 0..n-1.
    """
    cfg = config or TransitionDetectionConfig()
    raw = pd.Series(np.asarray(signal, dtype=float))
    if index is not None:
        pos = pd.to_numeric(pd.Series(index), errors="coerce").values
    else:
        pos = np.arange(len(raw), dtype=float)

    df = pd.DataFrame({"_sig": raw.values, "_pos": pos})
    df = df.dropna(subset=["_sig"])
    if len(df) < cfg.min_history:
        return _empty_result(cfg, "insufficient_history_after_dropna", {})

    df = df.rename(columns={"_sig": "__signal__", "_pos": cfg.index_column})
    return detect_transition(
        df,
        cfg,
        signal_column_override="__signal__",
    )


def detect_transition(
    df: pd.DataFrame,
    config: TransitionDetectionConfig | None = None,
    *,
    signal_column_override: str | None = None,
) -> dict[str, Any]:
    """
    Detect first sustained structural transition on a single-entity timeseries.

    Parameters
    ----------
    df
        Time-ordered rows for one entity (unit, asset, ...).
    config
        :class:`TransitionDetectionConfig`; defaults used if None.
    signal_column_override
        Internal: use this column name as the primary signal (used by ``detect_transition_from_signal``).

    Returns
    -------
    dict
        Keys include ``detected``, ``transition_index``, ``transition_position``, ``detection_reason``,
        ``signal_used``, ``warmup_applied``, ``threshold_used``, ``diagnostics``, ``confidence``,
        and optionally ``normalized_progress_at_detection``, ``lead_time``, ``remaining_life_normalized``.
    """
    cfg = config or TransitionDetectionConfig()
    diagnostics: dict[str, Any] = {}

    if df is None or len(df) == 0:
        return _empty_result(cfg, "no_data", diagnostics)

    if cfg.index_column not in df.columns:
        return _empty_result(cfg, f"missing_index:{cfg.index_column}", diagnostics)

    d0, warmup_meta = _apply_warmup_mask(df, cfg)
    diagnostics["warmup"] = warmup_meta

    if len(d0) < cfg.min_history:
        return _empty_result(cfg, "insufficient_post_warmup", diagnostics, warmup_meta)

    signal_col: str | None
    signal_label: str

    if signal_column_override:
        signal_col = signal_column_override
        signal_label = signal_col
    elif cfg.signal_mode == "composite":
        comp, signal_label = _build_composite_signal(d0, cfg)
        if comp is None:
            return _empty_result(cfg, "composite_unavailable", diagnostics)
        d0 = d0.copy()
        d0["__composite__"] = comp.values
        signal_col = "__composite__"
    else:
        sc, tried = _select_signal_column(d0, cfg)
        diagnostics["signal_candidates_tried"] = tried
        if sc is None:
            return _empty_result(cfg, "no_signal_column", diagnostics)
        signal_col = sc
        signal_label = sc

    raw = _as_float_series(d0[signal_col])
    if raw.isna().all() or float(raw.std(skipna=True) or 0.0) < EPS:
        return _empty_result(cfg, "flat_or_invalid_signal", diagnostics)

    if cfg.use_relative_thresholds:
        norm = _normalize_01(raw)
    else:
        norm = raw.astype(float)

    smooth = norm.rolling(window=int(cfg.smoothing_window), min_periods=1).mean()
    positions = pd.to_numeric(d0[cfg.index_column], errors="coerce").values
    sm = smooth.values.astype(float)
    n = len(sm)

    thr = float(cfg.threshold)
    threshold_used = thr

    sustained = max(1, int(cfg.sustained_points))
    t_idx: int | None = None
    reason = "threshold_crossing"

    run = 0
    for i in range(n):
        if sm[i] > thr:
            run += 1
            if run >= sustained:
                t_idx = i - sustained + 1
                break
        else:
            run = 0

    if t_idx is None and cfg.fallback_mode == "max_slope":
        d1 = np.diff(sm, prepend=sm[0])
        d1_smooth = pd.Series(d1).rolling(max(2, min(5, cfg.smoothing_window)), min_periods=1).mean().values
        search_from = min(3, max(0, n - 1))
        if n > search_from + 1:
            seg = d1_smooth[search_from:]
            if not np.all(np.isnan(seg)):
                t_idx = int(np.nanargmax(seg)) + search_from
                reason = "max_slope_increase"

    if t_idx is None and cfg.fallback_mode == "regime_shift":
        d2 = np.abs(np.diff(sm, n=2, prepend=np.r_[sm[0], sm[0]]))
        if len(d2) > 4:
            t_idx = int(np.nanargmax(d2[2:-2])) + 2 if not np.all(np.isnan(d2)) else None
            if t_idx is not None:
                reason = "regime_shift"

    if t_idx is None or t_idx < 0 or t_idx >= n:
        if cfg.verbose:
            print(f"[transition_detection] no_detection reason={cfg.fallback_mode} thr={thr} n={n}")
        return _empty_result(cfg, "no_detection", diagnostics, warmup_meta)

    pos_sel = float(positions[t_idx])
    pos_max = float(np.nanmax(positions))
    pos_min = float(np.nanmin(positions))
    span_pos = max(EPS, pos_max - pos_min)
    norm_progress = float(np.clip((pos_sel - pos_min) / span_pos, 0.0, 1.0))
    lead = max(0.0, pos_max - pos_sel)
    rln = float(np.clip(lead / span_pos, 0.0, 1.0))

    margin = float(sm[t_idx] - thr) if thr <= 1.0 else 0.5
    confidence = float(np.clip(0.35 + 0.65 * min(1.0, max(0.0, margin / 0.5)), 0.0, 1.0))

    if cfg.require_monotonic_confirmation and t_idx > 0 and sm[t_idx] < sm[t_idx - 1]:
        confidence *= 0.85

    if cfg.ensemble_agreement and cfg.signal_mode == "single":
        agreement = _signal_agreement_score(d0, cfg, t_idx)
        diagnostics["ensemble_agreement"] = agreement
        confidence = float(np.clip(confidence * (0.7 + 0.3 * agreement), 0.0, 1.0))

    diagnostics.update(
        {
            "signal_min": float(raw.min()),
            "signal_max": float(raw.max()),
            "smoothed_at_transition": float(sm[t_idx]),
            "index_max": pos_max,
        }
    )

    if cfg.verbose:
        print(
            f"[transition_detection] detected idx={t_idx} pos={pos_sel} signal={signal_label} reason={reason} conf={confidence:.3f}"
        )

    return {
        "detected": True,
        "transition_index": int(t_idx),
        "transition_position": pos_sel,
        "detection_reason": reason,
        "signal_used": signal_label,
        "warmup_applied": warmup_meta,
        "threshold_used": threshold_used,
        "diagnostics": diagnostics,
        "confidence": confidence,
        "normalized_progress_at_detection": norm_progress,
        "lead_time": lead,
        "remaining_life_normalized": rln,
        "transition_cycle": pos_sel,
        "remaining_life_normalized_legacy": rln,
        "lead_time_cycles": lead,
        "reason": reason,
        "metadata": {**warmup_meta, **diagnostics},
    }


def _signal_agreement_score(df: pd.DataFrame, cfg: TransitionDetectionConfig, ref_idx: int, window: int = 3) -> float:
    """Fraction of available priority signals that show elevated values near ref_idx."""
    hits = 0
    total = 0
    for col in cfg.signal_priority:
        if col not in df.columns or col == cfg.primary_signal:
            continue
        raw = _as_float_series(df[col])
        if raw.std(skipna=True) < EPS:
            continue
        nrm = _normalize_01(raw)
        sm = nrm.rolling(window=int(cfg.smoothing_window), min_periods=1).mean().values
        lo = max(0, ref_idx - window)
        hi = min(len(sm), ref_idx + window + 1)
        seg = sm[lo:hi]
        if len(seg) == 0:
            continue
        total += 1
        if float(np.max(seg)) > float(cfg.relative_threshold_low):
            hits += 1
    return float(hits / total) if total else 0.5


def _empty_result(
    cfg: TransitionDetectionConfig,
    reason: str,
    diagnostics: dict[str, Any],
    warmup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "detected": False,
        "transition_index": None,
        "transition_position": None,
        "detection_reason": reason,
        "signal_used": None,
        "warmup_applied": warmup or {},
        "threshold_used": float(cfg.threshold),
        "diagnostics": diagnostics,
        "confidence": 0.0,
        "normalized_progress_at_detection": None,
        "lead_time": None,
        "remaining_life_normalized": None,
        "transition_cycle": None,
        "remaining_life_normalized_legacy": None,
        "lead_time_cycles": None,
        "reason": reason,
        "metadata": diagnostics,
    }


def summarize_transition_detection(
    df: pd.DataFrame,
    config: TransitionDetectionConfig | None = None,
    *,
    group_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run per-group detection and aggregate metrics.

    ``group_column`` overrides :attr:`TransitionDetectionConfig.entity_group_column` when provided.
    """
    from neraium_core.detection.transition_evaluation import summarize_entity_transitions

    cfg = config or TransitionDetectionConfig()
    gc = group_column if group_column is not None else cfg.entity_group_column
    return summarize_entity_transitions(df, gc, config=cfg)
