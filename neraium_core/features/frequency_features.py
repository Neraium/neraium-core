from __future__ import annotations

from typing import Iterable

import numpy as np

_EPS = 1e-12


def _safe_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros(1, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def power_spectrum(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    x = _safe_array(values)
    n = int(x.size)
    if n < 2:
        return np.array([0.0], dtype=float), np.array([0.0], dtype=float)
    centered = x - float(np.mean(x))
    fft = np.fft.rfft(centered)
    power = np.square(np.abs(fft)) / float(max(1, n))
    freqs = np.fft.rfftfreq(n, d=1.0)
    return freqs.astype(float), power.astype(float)


def band_energy(values: Iterable[float], bands: list[tuple[float, float]] | None = None) -> dict[str, float]:
    if bands is None:
        bands = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5)]
    freqs, power = power_spectrum(values)
    total = float(np.sum(power)) + _EPS
    out: dict[str, float] = {}
    for low, high in bands:
        mask = (freqs >= float(low)) & (freqs < float(high))
        val = float(np.sum(power[mask]) / total)
        key = f"band_energy_{low:.2f}_{high:.2f}".replace(".", "p")
        out[key] = val
    return out


def dominant_frequency_magnitude(values: Iterable[float]) -> float:
    _freqs, power = power_spectrum(values)
    if power.size <= 1:
        return 0.0
    idx = int(np.argmax(power[1:]) + 1)
    return float(power[idx])


def spectral_centroid(values: Iterable[float]) -> float:
    freqs, power = power_spectrum(values)
    den = float(np.sum(power)) + _EPS
    return float(np.sum(freqs * power) / den)


def spectral_spread(values: Iterable[float]) -> float:
    freqs, power = power_spectrum(values)
    centroid = spectral_centroid(values)
    den = float(np.sum(power)) + _EPS
    var = float(np.sum(np.square(freqs - centroid) * power) / den)
    return float(np.sqrt(max(0.0, var)))


def spectral_entropy(values: Iterable[float]) -> float:
    _freqs, power = power_spectrum(values)
    p = power / (float(np.sum(power)) + _EPS)
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    ent = -float(np.sum(p * np.log2(p + _EPS)))
    max_ent = float(np.log2(max(2, p.size)))
    return float(max(0.0, min(1.0, ent / (max_ent + _EPS))))


def dominant_frequency_ratio(values: Iterable[float]) -> float:
    _freqs, power = power_spectrum(values)
    total = float(np.sum(power)) + _EPS
    if power.size <= 1:
        return 0.0
    idx = int(np.argmax(power[1:]) + 1)
    return float(power[idx] / total)


def low_high_frequency_energy_ratio(values: Iterable[float], split: float = 0.2) -> float:
    freqs, power = power_spectrum(values)
    low = float(np.sum(power[freqs < split]))
    high = float(np.sum(power[freqs >= split]))
    return float(low / (high + _EPS))


def spectral_energy_dispersion(values: Iterable[float]) -> float:
    _freqs, power = power_spectrum(values)
    p = power / (float(np.sum(power)) + _EPS)
    return float(np.std(p))


def spectral_energy_concentration(values: Iterable[float], top_k: int = 3) -> float:
    _freqs, power = power_spectrum(values)
    total = float(np.sum(power)) + _EPS
    if power.size == 0:
        return 0.0
    k = max(1, min(int(top_k), int(power.size)))
    top = np.sort(power)[-k:]
    return float(np.sum(top) / total)
