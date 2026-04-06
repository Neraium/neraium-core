"""False-positive suppression layer for AtomicMonitor.

This module adds conformal calibration, contextual gating, consensus checks,
FDR control, and surge/cooldown protection for production-oriented alert trust.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .atomic_alerting import AlertTransition
from .atomic_monitor import AtomicMonitor, AtomicUpdate


@dataclass
class CalibratedUpdate:
    alert_level: str
    raw_level: str
    suppressed: bool
    score: float
    components: Dict[str, float]
    validations: Dict[str, bool]
    localization: Dict[str, object]


class AtomicMonitorCalibrated(AtomicMonitor):
    """Atomic monitor with multi-layer false-positive suppression."""

    def __init__(self, sensors: List[str], config: dict) -> None:
        fp_defaults = {
            "conformal_enabled": True,
            "conformal_alpha": 0.05,
            "conformal_window": 1000,
            "operational_modes": None,
            "maintenance_windows": False,
            "diurnal_patterns": True,
            "consensus_required": 2,
            "consensus_window": 10,
            "threshold_adaptation": True,
            "target_fp_rate": 0.01,
            "fp_history_window": 1000,
            "bh_enabled": True,
            "bh_fdr_target": 0.05,
            "min_alert_duration": 5,
            "cooldown_period": 50,
        }
        merged = {**fp_defaults, **dict(config)}
        super().__init__(sensors, merged)
        self.fp_control: Dict[str, object] = {
            "detector_firing_history": np.empty((0, len(self.weights)), dtype=bool),
            "alert_cooldown_counter": 0,
            "consecutive_yellow": 0,
            "time_model": None,
        }
        self.calibration: Dict[str, object] = {
            "thresholds_learned": False,
            "null_distributions": {},
            "conformal_cutoffs": {},
        }
        self.suppression_log: List[Dict[str, object]] = []
        self.calibrated_alert_history: List[Dict[str, object]] = []

    def calibrate(
        self,
        healthy_data: np.ndarray,
        validation_data: Optional[np.ndarray] = None,
        n_bootstrap: int = 1000,
        verbose: bool = False,
    ) -> None:
        del n_bootstrap
        healthy_data = np.asarray(healthy_data, dtype=float)
        n = healthy_data.shape[0]
        if n < 30:
            self.learn_baseline(healthy_data)
            self.calibration["thresholds_learned"] = False
            return

        n_train = max(5, int(0.7 * n))
        train = healthy_data[:n_train]
        cal = healthy_data[n_train:]

        self.learn_baseline(train)
        self._reset_runtime_state()
        cal_scores = self._collect_calibration_scores(cal)
        self._reset_runtime_state()
        self.calibration["null_distributions"] = self._fit_null_distributions(cal_scores)
        if bool(self.config.get("conformal_enabled", True)):
            self.calibration["conformal_cutoffs"] = self._compute_conformal_cutoffs(cal_scores)
        if bool(self.config.get("diurnal_patterns", True)):
            self.fp_control["time_model"] = self._learn_temporal_patterns(cal)

        self.calibration["thresholds_learned"] = True

        if validation_data is not None and len(validation_data) > 0:
            fp_rate = self._validate_fp_rate(np.asarray(validation_data, dtype=float))
            if fp_rate > 2 * float(self.config.get("target_fp_rate", 0.01)):
                self._auto_adjust_thresholds(fp_rate)

        if verbose:
            print("Calibration complete.")

    def update(self, x_t: np.ndarray, timestamp: Optional[float] = None, context: Optional[dict] = None) -> CalibratedUpdate:  # type: ignore[override]
        context = context or {}
        cooldown = int(self.fp_control.get("alert_cooldown_counter", 0))
        if cooldown > 0:
            self.fp_control["alert_cooldown_counter"] = cooldown - 1

        prev_level = self.alert_machine.level
        prev_hist_len = len(self.alert_machine.history)

        raw = super().update(x_t, timestamp=timestamp)
        scores = raw.components

        conformal_valid = self._apply_conformal_calibration(scores)
        context_valid = self._apply_contextual_gating(context)
        consensus_valid = self._apply_consensus_check(scores)
        bh_valid = self._apply_bh_correction(scores)
        surge_valid = self._apply_surge_protection(prev_level, raw.alert_level)
        cooldown_open = int(self.fp_control.get("alert_cooldown_counter", 0)) == 0

        should_alert = all([conformal_valid, context_valid, consensus_valid, bh_valid, surge_valid, cooldown_open])

        final_level = raw.alert_level
        suppressed = False
        if not should_alert and raw.alert_level == "RED":
            suppressed = True
            final_level = "YELLOW"
            self._restore_alert_state(prev_level, prev_hist_len)
            if prev_level != final_level:
                t = float(self.counter if timestamp is None else timestamp)
                self.alert_machine.level = final_level
                self.alert_machine.history.append(
                    AlertTransition(
                        timestamp=t,
                        level=final_level,
                        score=float(raw.score),
                        dominant_detector=raw.dominant_detector,
                    )
                )
            self._log_suppression(raw.alert_level, {
                "conformal": not conformal_valid,
                "context": not context_valid,
                "consensus": not consensus_valid,
                "bh": not bh_valid,
                "surge": not surge_valid,
            })

        if should_alert and raw.alert_level == "RED" and prev_level != "RED":
            self.fp_control["alert_cooldown_counter"] = int(self.config.get("cooldown_period", 50))
            self.calibrated_alert_history.append(
                {
                    "timestamp": float(self.counter if timestamp is None else timestamp),
                    "level": "RED",
                    "score": float(raw.score),
                    "dominant_detector": raw.dominant_detector,
                    "context_mode": context.get("mode", "unknown"),
                    "calibrated": True,
                }
            )

        return CalibratedUpdate(
            alert_level=final_level,
            raw_level=raw.alert_level,
            suppressed=suppressed,
            score=float(raw.score),
            components=scores,
            validations={
                "conformal": conformal_valid,
                "context": context_valid,
                "consensus": consensus_valid,
                "bh": bh_valid,
                "surge": surge_valid,
            },
            localization=raw.localization,
        )

    def get_suppression_log(self, n: int = 20) -> List[Dict[str, object]]:
        return self.suppression_log[-n:]

    def get_fp_rate_estimate(self, window: int = 1000) -> float:
        if not self.calibrated_alert_history:
            return 0.0
        recent = self.calibrated_alert_history[-window:]
        red = sum(1 for r in recent if r["level"] == "RED")
        return red / max(len(recent), 1)

    def _reset_runtime_state(self) -> None:
        self.buffer = np.empty((0, len(self.sensors)), dtype=float)
        self.counter = 0
        self.prev_x = None
        self.component_history.clear()
        self.score_history.clear()
        self.alert_machine.level = "GREEN"
        self.alert_machine.history.clear()

    def _restore_alert_state(self, level: str, history_len: int) -> None:
        self.alert_machine.level = level
        if len(self.alert_machine.history) > history_len:
            del self.alert_machine.history[history_len:]

    def _collect_calibration_scores(self, cal_data: np.ndarray) -> Dict[str, np.ndarray]:
        store: Dict[str, List[float]] = {k: [] for k in self.weights}
        for row in cal_data:
            upd = super().update(row)
            for k in store:
                store[k].append(float(upd.components.get(k, 0.0)))
        return {k: np.asarray(v, dtype=float) for k, v in store.items()}

    def _fit_null_distributions(self, scores: Dict[str, np.ndarray]) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        for name, vals in scores.items():
            s = vals[np.isfinite(vals)]
            if s.size < 10:
                continue
            q = np.quantile(s, np.linspace(0, 1, 501))
            out[name] = {"quantiles": q, "ecdf": np.sort(s)}
        return out

    def _compute_conformal_cutoffs(self, scores: Dict[str, np.ndarray]) -> Dict[str, float]:
        alpha = float(self.config.get("conformal_alpha", 0.05))
        out: Dict[str, float] = {}
        for name, vals in scores.items():
            s = vals[vals > 0]
            if s.size > 5:
                out[name] = float(np.quantile(s, 1 - alpha))
        return out

    def _learn_temporal_patterns(self, data: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
        hours = np.arange(data.shape[0]) % 24
        model: Dict[int, Dict[str, np.ndarray]] = {}
        for h in range(24):
            idx = np.where(hours == h)[0]
            if idx.size < 5:
                continue
            block = data[idx]
            model[h] = {
                "mean": block.mean(axis=0),
                "cov": np.cov(block.T) + 1e-6 * np.eye(block.shape[1]),
            }
        return model

    def _validate_fp_rate(self, data: np.ndarray) -> float:
        red = 0
        total = 0
        for row in data:
            upd = self.update(row, context={})
            total += 1
            if upd.alert_level == "RED":
                red += 1
        return red / max(total, 1)

    def _auto_adjust_thresholds(self, current_fp_rate: float) -> None:
        if not bool(self.config.get("threshold_adaptation", True)):
            return
        target = float(self.config.get("target_fp_rate", 0.01))
        ratio = max(current_fp_rate / max(target, 1e-6), 1.0)
        grow = float(np.sqrt(ratio))
        self.alert_machine.green_yellow = min(self.alert_machine.green_yellow * grow, 0.9)
        self.alert_machine.yellow_red = min(self.alert_machine.yellow_red * grow, 0.99)

    def _apply_conformal_calibration(self, scores: Dict[str, float]) -> bool:
        if not bool(self.config.get("conformal_enabled", True)):
            return True
        if not bool(self.calibration.get("thresholds_learned", False)):
            return True
        cutoffs: Dict[str, float] = self.calibration.get("conformal_cutoffs", {})  # type: ignore[assignment]
        violations = 0
        for name, value in scores.items():
            cutoff = cutoffs.get(name)
            if cutoff is not None and value > cutoff:
                violations += 1
        return violations >= int(self.config.get("consensus_required", 2))

    def _apply_contextual_gating(self, context: dict) -> bool:
        if bool(context.get("maintenance", False)):
            return False
        if bool(self.config.get("diurnal_patterns", True)) and self.fp_control.get("time_model") and "hour" in context:
            hour = int(context["hour"])
            if hour not in self.fp_control["time_model"]:  # type: ignore[operator]
                return True
        return True

    def _apply_consensus_check(self, scores: Dict[str, float]) -> bool:
        keys = list(self.weights)
        firing = np.array([scores.get(k, 0.0) > self.alert_machine.green_yellow for k in keys], dtype=bool)
        hist = self.fp_control["detector_firing_history"]
        hist = np.vstack([hist, firing])
        window = int(self.config.get("consensus_window", 10))
        if hist.shape[0] > window:
            hist = hist[-window:]
        self.fp_control["detector_firing_history"] = hist
        if hist.shape[0] < window:
            return False
        persistent = np.sum(hist, axis=0) >= int(np.ceil(0.6 * window))
        return int(np.sum(persistent)) >= int(self.config.get("consensus_required", 2))

    def _apply_bh_correction(self, scores: Dict[str, float]) -> bool:
        if not bool(self.config.get("bh_enabled", True)):
            return True
        nulls = self.calibration.get("null_distributions", {})
        if not nulls:
            return True

        pvals = []
        names = []
        for name, score in scores.items():
            nd = nulls.get(name)
            if not nd:
                continue
            ecdf = nd["ecdf"]
            rank = np.searchsorted(ecdf, score, side="right")
            p = 1.0 - rank / max(len(ecdf), 1)
            pvals.append(float(np.clip(p, 0.0, 1.0)))
            names.append(name)
        if not pvals:
            return True

        pvals = np.asarray(pvals)
        order = np.argsort(pvals)
        sorted_p = pvals[order]
        m = len(sorted_p)
        alpha = float(self.config.get("bh_fdr_target", 0.05))
        thresh = (np.arange(1, m + 1) / m) * alpha
        reject = sorted_p <= thresh
        if not np.any(reject):
            return False
        k = int(np.max(np.where(reject)[0])) + 1
        sig = {names[order[i]] for i in range(k)}
        top2 = set(sorted(scores, key=scores.get, reverse=True)[:2])
        return len(sig & top2) > 0

    def _apply_surge_protection(self, prior_level: str, proposed_level: str) -> bool:
        if prior_level == "YELLOW":
            self.fp_control["consecutive_yellow"] = int(self.fp_control.get("consecutive_yellow", 0)) + 1
        else:
            self.fp_control["consecutive_yellow"] = 0
        if proposed_level == "RED":
            return int(self.fp_control.get("consecutive_yellow", 0)) >= int(self.config.get("min_alert_duration", 5))
        return True

    def _log_suppression(self, raw_level: str, reasons: Dict[str, bool]) -> None:
        self.suppression_log.append(
            {
                "timestamp": float(self.counter),
                "raw_level": raw_level,
                "suppressed_to": "YELLOW",
                "reasons": [k for k, v in reasons.items() if v],
            }
        )
