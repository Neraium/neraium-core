from __future__ import annotations

import math
from typing import Iterable

import numpy as np

_EPS = 1e-12


def _safe_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros(1, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def rms(values: Iterable[float]) -> float:
    x = _safe_array(values)
    return float(np.sqrt(np.mean(np.square(x))))


def peak_to_peak(values: Iterable[float]) -> float:
    x = _safe_array(values)
    return float(np.max(x) - np.min(x))


def variance(values: Iterable[float]) -> float:
    x = _safe_array(values)
    return float(np.var(x))


def skewness(values: Iterable[float]) -> float:
    x = _safe_array(values)
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma <= _EPS:
        return 0.0
    z = (x - mu) / sigma
    return float(np.mean(np.power(z, 3)))


def kurtosis(values: Iterable[float]) -> float:
    x = _safe_array(values)
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma <= _EPS:
        return 0.0
    z = (x - mu) / sigma
    return float(np.mean(np.power(z, 4)))


def crest_factor(values: Iterable[float]) -> float:
    x = _safe_array(values)
    den = rms(x)
    if den <= _EPS:
        return 0.0
    return float(np.max(np.abs(x)) / den)


def mean_absolute_deviation(values: Iterable[float]) -> float:
    x = _safe_array(values)
    mu = float(np.mean(x))
    return float(np.mean(np.abs(x - mu)))


def rolling_energy(values: Iterable[float], window: int = 8) -> float:
    x = _safe_array(values)
    if x.size <= 1:
        return float(np.mean(np.square(x)))
    w = max(2, min(int(window), int(x.size)))
    energies = np.square(x)
    kernel = np.ones(w, dtype=float) / float(w)
    smoothed = np.convolve(energies, kernel, mode="valid")
    return float(np.mean(smoothed))


def shannon_entropy(values: Iterable[float], bins: int = 16) -> float:
    x = _safe_array(values)
    if np.allclose(x, x[0]):
        return 0.0
    b = max(4, int(bins))
    counts, _ = np.histogram(x, bins=b)
    probs = counts.astype(float)
    probs = probs / (float(np.sum(probs)) + _EPS)
    probs = probs[probs > 0.0]
    if probs.size == 0:
        return 0.0
    ent = -float(np.sum(probs * np.log2(probs + _EPS)))
    max_ent = math.log2(float(b)) + _EPS
    return float(max(0.0, min(1.0, ent / max_ent)))


def zero_crossing_rate(values: Iterable[float]) -> float:
    x = _safe_array(values)
    if x.size < 2:
        return 0.0
    signs = np.sign(x)
    changes = np.sum(np.abs(np.diff(signs)) > 0)
    return float(changes / max(1, x.size - 1))


def lag1_autocorrelation(values: Iterable[float]) -> float:
    x = _safe_array(values)
    if x.size < 3:
        return 0.0
    a = x[:-1] - float(np.mean(x[:-1]))
    b = x[1:] - float(np.mean(x[1:]))
    den = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if den <= _EPS:
        return 0.0
    return float(np.dot(a, b) / den)


def trend_slope(values: Iterable[float]) -> float:
    x = _safe_array(values)
    n = int(x.size)
    if n < 3:
        return 0.0
    t = np.arange(n, dtype=float)
    tc = t - float(np.mean(t))
    xc = x - float(np.mean(x))
    den = float(np.dot(tc, tc)) + _EPS
    return float(np.dot(tc, xc) / den)


def smoothness(values: Iterable[float]) -> float:
    x = _safe_array(values)
    if x.size < 3:
        return 1.0
    d2 = np.diff(x, n=2)
    rough = float(np.mean(np.abs(d2)))
    scale = float(np.mean(np.abs(x))) + 1.0
    return float(1.0 / (1.0 + rough / scale))


def roughness(values: Iterable[float]) -> float:
    return float(1.0 - smoothness(values))


def envelope_mean(values: Iterable[float], window: int = 8) -> float:
    x = _safe_array(values)
    w = max(2, min(int(window), int(x.size)))
    env = np.convolve(np.abs(x), np.ones(w) / float(w), mode="valid")
    return float(np.mean(env)) if env.size else float(np.mean(np.abs(x)))


def envelope_std(values: Iterable[float], window: int = 8) -> float:
    x = _safe_array(values)
    w = max(2, min(int(window), int(x.size)))
    env = np.convolve(np.abs(x), np.ones(w) / float(w), mode="valid")
    return float(np.std(env)) if env.size else 0.0
