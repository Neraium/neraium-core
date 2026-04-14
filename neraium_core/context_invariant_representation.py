from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


REFERENCE_STRATEGIES = {"initial_window", "rolling", "ewma", "robust"}
REPRESENTATION_MODES = {"raw", "residual", "dynamic", "combined"}


def _sliding_row_windows(a: np.ndarray, w: int) -> np.ndarray:
    """Shape (n - w + 1, w, d); 1D sliding per column avoids NumPy 2.x axis quirks for d == 1."""
    n, d = a.shape
    if w > n:
        return np.zeros((0, w, d), dtype=np.float64)
    if d == 1:
        sw = sliding_window_view(np.ascontiguousarray(a[:, 0]), w)
        return np.ascontiguousarray(sw[:, :, np.newaxis])
    cols = [sliding_window_view(np.ascontiguousarray(a[:, j]), w) for j in range(d)]
    return np.stack(cols, axis=-1)


@dataclass(frozen=True)
class RepresentationWeights:
    raw_weight: float = 0.35
    residual_weight: float = 1.2
    delta_weight: float = 1.35
    slope_weight: float = 1.1
    drift_weight: float = 1.2
    second_diff_weight: float = 0.75


@dataclass(frozen=True)
class TemporalRepresentationConfig:
    mode: str = "combined"
    reference_strategy: str = "robust"
    reference_window: int = 24
    local_window: int = 8
    slope_window: int = 6
    ewma_alpha: float = 0.2
    enable_diagnostics: bool = True
    weights: RepresentationWeights = RepresentationWeights()

    def resolved_mode(self) -> str:
        m = str(self.mode or "combined").strip().lower()
        return m if m in REPRESENTATION_MODES else "combined"

    def resolved_strategy(self) -> str:
        s = str(self.reference_strategy or "robust").strip().lower()
        return s if s in REFERENCE_STRATEGIES else "robust"


@dataclass(frozen=True)
class TemporalRepresentation:
    transformed: np.ndarray
    feature_names: list[str]
    diagnostics: dict[str, float | bool | str]


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean with min(t+1, w) observations at row t (same semantics as the legacy loop).

    Vectorized via prefix cumsums: O(n·d) vs O(n·w·d) for the naive implementation.
    """
    n, d = arr.shape
    w = max(1, int(window))
    a = np.asarray(arr, dtype=float)
    if w == 1:
        return a
    c = np.cumsum(np.concatenate([np.zeros((1, d), dtype=float), a], axis=0), axis=0)
    out = np.zeros_like(a, dtype=float)
    wm1 = w - 1
    if wm1 > 0:
        for t in range(min(wm1, n)):
            out[t] = c[t + 1] / float(t + 1)
    if n > wm1:
        out[wm1:] = (c[w : n + 1] - c[0 : n - wm1]) / float(w)
    return out


def _rolling_median(arr: np.ndarray, window: int) -> np.ndarray:
    """Efficient rolling median using sorted list per column for O(w·log w) complexity.

    Instead of recomputing full median via np.median (which sorts the window each time),
    maintain a sorted list and insert/remove in O(log w) via bisect. This gives O(n·d·log w)
    instead of O(n·d·w·log w).
    """
    n, d = arr.shape
    w = max(1, int(window))
    a = np.asarray(arr, dtype=float)
    out = np.zeros_like(a, dtype=float)

    if w == 1:
        return a

    # For each column, maintain a sorted list and compute rolling median
    sorted_windows = [[] for _ in range(d)]
    deque_windows = [[] for _ in range(d)]

    for t in range(n):
        lo = max(0, t - w + 1)
        start_idx = max(0, t - w + 1)
        end_idx = t + 1

        # For initial partial windows, rebuild from scratch (smaller cost)
        if t < w - 1:
            # Rebuild sorted window for this time step
            window_vals = a[start_idx:end_idx]
            for col in range(d):
                sorted_windows[col] = sorted(window_vals[:, col])
                deque_windows[col] = list(window_vals[:, col])
        else:
            # Sliding window: remove oldest, add newest
            old_val = a[t - w, :]
            new_val = a[t, :]

            for col in range(d):
                # Remove old value from sorted list
                old_v = old_val[col]
                sorted_windows[col].pop(bisect.bisect_left(sorted_windows[col], old_v))
                # Add new value
                new_v = new_val[col]
                bisect.insort(sorted_windows[col], new_v)

        # Extract median from sorted list
        sorted_len = len(sorted_windows[0])
        mid = sorted_len // 2
        for col in range(d):
            if sorted_len % 2 == 1:
                out[t, col] = sorted_windows[col][mid]
            else:
                out[t, col] = (sorted_windows[col][mid - 1] + sorted_windows[col][mid]) / 2.0

    return out


def _ewma(arr: np.ndarray, alpha: float) -> np.ndarray:
    n, _ = arr.shape
    a = float(max(1e-3, min(1.0, alpha)))
    out = np.zeros_like(arr, dtype=float)
    out[0] = arr[0]
    for t in range(1, n):
        out[t] = a * arr[t] + (1.0 - a) * out[t - 1]
    return out


def _slope(arr: np.ndarray, window: int, timestamps: np.ndarray | None = None) -> np.ndarray:
    n, d = arr.shape
    w = max(2, int(window))
    out = np.zeros((n, d), dtype=float)
    a = np.asarray(arr, dtype=float)

    if timestamps is None:
        wm1 = w - 1
        for t in range(min(wm1, n)):
            lo = max(0, t - w + 1)
            y = a[lo : t + 1]
            if y.shape[0] < 2:
                continue
            x = np.arange(y.shape[0], dtype=float)
            x = x - float(np.mean(x))
            denom = float(np.sum(x * x))
            if denom <= 1e-12:
                continue
            y_centered = y - np.mean(y, axis=0)
            out[t] = (x[:, None] * y_centered).sum(axis=0) / denom
        if n > wm1:
            sw = _sliding_row_windows(a, w)
            xc = np.arange(w, dtype=float)
            xc = xc - float(np.mean(xc))
            denom = float(np.dot(xc, xc))
            if denom > 1e-12:
                yc = sw - sw.mean(axis=1, keepdims=True)
                out[wm1:] = (yc * xc[np.newaxis, :, np.newaxis]).sum(axis=1) / denom
        return out

    for t in range(n):
        lo = max(0, t - w + 1)
        y = a[lo : t + 1]
        if y.shape[0] < 2:
            continue
        x = np.asarray(timestamps[lo : t + 1], dtype=float)
        x = x - float(np.mean(x))
        denom = float(np.sum(x * x))
        if denom <= 1e-12:
            continue
        y_centered = y - np.mean(y, axis=0)
        out[t] = (x[:, None] * y_centered).sum(axis=0) / denom
    return out


def _mad_scale(arr: np.ndarray) -> np.ndarray:
    med = np.median(arr, axis=0)
    mad = np.median(np.abs(arr - med), axis=0)
    return np.maximum(1e-6, 1.4826 * mad)


def _reference_series(arr: np.ndarray, *, strategy: str, window: int, ewma_alpha: float) -> np.ndarray:
    n, _ = arr.shape
    if n == 0:
        return arr
    w = max(2, int(window))
    if strategy == "initial_window":
        ref = np.mean(arr[: min(n, w)], axis=0)
        return np.repeat(ref[None, :], repeats=n, axis=0)
    if strategy == "rolling":
        return _rolling_mean(arr, w)
    if strategy == "ewma":
        return _ewma(arr, ewma_alpha)
    return _rolling_median(arr, w)


def build_temporal_representation(
    history: np.ndarray,
    config: TemporalRepresentationConfig,
    timestamps: np.ndarray | None = None,
) -> TemporalRepresentation:
    h = np.asarray(history, dtype=float)
    if h.ndim != 2 or h.size == 0:
        return TemporalRepresentation(
            transformed=np.asarray(h, dtype=float),
            feature_names=[],
            diagnostics={
                "context_dominance_score": 0.0,
                "dynamic_signal_strength": 0.0,
                "early_window_separability": 0.0,
                "trajectory_onset_divergence_timing": 0.0,
                "early_separation_flag": False,
                "representation_mode": config.resolved_mode(),
                "reference_strategy": config.resolved_strategy(),
            },
        )

    ts = None
    if timestamps is not None:
        t_arr = np.asarray(timestamps, dtype=float)
        if t_arr.shape[0] == h.shape[0]:
            ts = t_arr

    local_mean = _rolling_mean(h, config.local_window)
    reference = _reference_series(
        h,
        strategy=config.resolved_strategy(),
        window=config.reference_window,
        ewma_alpha=config.ewma_alpha,
    )
    if ts is None:
        dt = np.ones(max(1, h.shape[0] - 1), dtype=float)
    else:
        dt = np.maximum(1e-6, np.diff(ts))
    raw_delta = np.diff(h, axis=0)
    delta = np.vstack([np.zeros((1, h.shape[1]), dtype=float), raw_delta])
    rate = np.vstack([np.zeros((1, h.shape[1]), dtype=float), raw_delta / dt[:, None]])
    second_diff = np.vstack([np.zeros((1, h.shape[1]), dtype=float), np.diff(delta, axis=0)])
    slope = _slope(h, config.slope_window, timestamps=ts)

    residual = h - local_mean
    drift = h - reference

    scale = _mad_scale(h)
    residual_n = residual / scale
    drift_n = drift / scale
    delta_n = delta / scale
    second_diff_n = second_diff / scale
    slope_n = slope / scale
    rate_n = rate / scale

    w = config.weights
    mode = config.resolved_mode()

    n_features = h.shape[1]
    base_names = [f"signal_{i}" for i in range(n_features)]

    if mode == "raw":
        transformed = np.asarray(h, dtype=float)
        feature_names = list(base_names)
    elif mode == "residual":
        transformed = np.concatenate([w.residual_weight * residual_n, w.drift_weight * drift_n], axis=1)
        feature_names = [f"{n}__residual" for n in base_names] + [f"{n}__drift" for n in base_names]
    elif mode == "dynamic":
        transformed = np.concatenate(
            [
                w.residual_weight * residual_n,
                w.delta_weight * delta_n,
                w.slope_weight * slope_n,
                w.slope_weight * rate_n,
                w.drift_weight * drift_n,
                w.second_diff_weight * second_diff_n,
            ],
            axis=1,
        )
        feature_names = (
            [f"{n}__residual" for n in base_names]
            + [f"{n}__delta" for n in base_names]
            + [f"{n}__slope" for n in base_names]
            + [f"{n}__dx_dt" for n in base_names]
            + [f"{n}__drift" for n in base_names]
            + [f"{n}__second_diff" for n in base_names]
        )
    else:
        transformed = np.concatenate(
            [
                w.raw_weight * h,
                w.residual_weight * residual_n,
                w.delta_weight * delta_n,
                w.slope_weight * slope_n,
                w.slope_weight * rate_n,
                w.drift_weight * drift_n,
                w.second_diff_weight * second_diff_n,
            ],
            axis=1,
        )
        feature_names = (
            list(base_names)
            + [f"{n}__residual" for n in base_names]
            + [f"{n}__delta" for n in base_names]
            + [f"{n}__slope" for n in base_names]
            + [f"{n}__dx_dt" for n in base_names]
            + [f"{n}__drift" for n in base_names]
            + [f"{n}__second_diff" for n in base_names]
        )

    static_base = np.mean(np.abs(drift_n), axis=1)
    dynamic_base = np.mean(
        np.abs(np.concatenate([residual_n, delta_n, slope_n, rate_n, second_diff_n], axis=1)),
        axis=1,
    )
    if mode == "raw":
        static_offset = static_base * max(1.0, float(w.raw_weight))
        dynamic_activity = dynamic_base * 0.15
    elif mode == "residual":
        static_offset = static_base * 0.55
        dynamic_activity = dynamic_base * max(1.0, 0.6 * float(w.residual_weight + w.drift_weight))
    elif mode == "dynamic":
        static_offset = static_base * 0.3
        dynamic_activity = dynamic_base * max(1.0, float(w.delta_weight + w.slope_weight + w.second_diff_weight))
    else:
        static_offset = static_base * max(0.25, float(w.raw_weight))
        dynamic_activity = dynamic_base * max(1.0, float(w.residual_weight + w.delta_weight + w.slope_weight + w.drift_weight))
    early_n = max(1, int(np.ceil(h.shape[0] * 0.2)))
    late_n = max(1, int(np.ceil(h.shape[0] * 0.2)))
    early_sep = float(np.mean(static_offset[:early_n]))
    late_dyn = float(np.mean(dynamic_activity[-late_n:]))
    denom = early_sep + late_dyn + 1e-6
    context_dominance = float(early_sep / denom)
    dynamic_strength = float(late_dyn / denom)

    dyn_trace = np.asarray(dynamic_activity, dtype=float)
    thresh = float(np.percentile(dyn_trace, 65.0)) if dyn_trace.size else 0.0
    onset_idx = int(np.argmax(dyn_trace >= thresh)) if dyn_trace.size else 0
    onset_timing = float(onset_idx / max(1, dyn_trace.size - 1)) if dyn_trace.size else 0.0
    early_separation_flag = bool(context_dominance > 0.62 and onset_timing > 0.35)

    dt_cv = float(np.std(dt) / (np.mean(dt) + 1e-9)) if dt.size else 0.0

    diagnostics: dict[str, float | bool | str] = {
        "context_dominance_score": round(context_dominance, 6),
        "dynamic_signal_strength": round(dynamic_strength, 6),
        "early_window_separability": round(early_sep, 6),
        "trajectory_onset_divergence_timing": round(onset_timing, 6),
        "early_separation_flag": early_separation_flag,
        "delta_t_stability": round(float(max(0.0, min(1.0, 1.0 - dt_cv))), 6),
        "delta_t_irregularity": round(float(max(0.0, min(1.0, dt_cv))), 6),
        "representation_mode": mode,
        "reference_strategy": config.resolved_strategy(),
    }

    return TemporalRepresentation(transformed=transformed, feature_names=feature_names, diagnostics=diagnostics)
