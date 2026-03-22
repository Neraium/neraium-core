from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


DEFAULT_MIN_SAMPLES = 28
DEFAULT_WATCH_QUANTILE = 82.0
DEFAULT_ALERT_QUANTILE = 93.5


def calibrate_quantile_thresholds(
    samples: Iterable[float],
    *,
    watch_quantile: float = DEFAULT_WATCH_QUANTILE,
    alert_quantile: float = DEFAULT_ALERT_QUANTILE,
) -> tuple[float, float]:
    arr = np.asarray(list(samples), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0

    watch_thr = float(np.percentile(arr, watch_quantile))
    alert_thr = float(np.percentile(arr, alert_quantile))
    if alert_thr < watch_thr:
        watch_thr, alert_thr = alert_thr, watch_thr
    return watch_thr, alert_thr


class QuantileThresholdCalibrator:
    """
    Small helper for stable threshold calibration from nominal history.
    """

    def __init__(
        self,
        *,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        watch_quantile: float = DEFAULT_WATCH_QUANTILE,
        alert_quantile: float = DEFAULT_ALERT_QUANTILE,
        maxlen: int = 256,
    ) -> None:
        self.min_samples = int(min_samples)
        self.watch_quantile = float(watch_quantile)
        self.alert_quantile = float(alert_quantile)
        self.samples: deque[float] = deque(maxlen=maxlen)
        self._thresholds: tuple[float, float] | None = None

    @property
    def thresholds(self) -> tuple[float, float] | None:
        return self._thresholds

    def add(self, value: float) -> tuple[float, float] | None:
        self.samples.append(float(value))
        if self._thresholds is None and len(self.samples) >= self.min_samples:
            self._thresholds = calibrate_quantile_thresholds(
                self.samples,
                watch_quantile=self.watch_quantile,
                alert_quantile=self.alert_quantile,
            )
        return self._thresholds
