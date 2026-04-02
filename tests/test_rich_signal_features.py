from __future__ import annotations

import numpy as np

from neraium_core.alignment import StructuralEngine
from neraium_core.features.window_feature_extractor import extract_window_features


def test_extract_window_features_returns_rich_and_cross_channel_features() -> None:
    t = np.linspace(0.0, 6.28, 128)
    window = np.stack(
        [
            np.sin(t),
            0.7 * np.sin(2.0 * t + 0.3),
            0.2 * np.cos(0.5 * t),
        ],
        axis=1,
    )

    payload = extract_window_features(window, channel_names=["a", "b", "c"])
    assert "a__rms" in payload.flattened
    assert "a__spectral_entropy" in payload.flattened
    assert "cross__coherence_like" in payload.flattened
    assert payload.cross_channel["coherence_like"] >= 0.0


def test_engine_populates_signal_degradation_on_multichannel_stream() -> None:
    engine = StructuralEngine(baseline_window=8, recent_window=4)
    outputs = []
    for i in range(20):
        frame = {
            "timestamp": f"2026-01-01T00:00:{i:02d}+00:00",
            "site_id": "site-x",
            "asset_id": "asset-x",
            "sensor_values": {
                "s1": float(np.sin(i / 3.0) + 0.02 * i),
                "s2": float(np.cos(i / 4.0) + (0.03 * i if i > 10 else 0.0)),
                "s3": float(np.sin(i / 2.0) * 0.5),
            },
        }
        outputs.append(engine.process_frame(frame))

    latest = outputs[-1]
    assert "signal_degradation" in latest
    sd = latest["signal_degradation"]
    assert "energy_instability_score" in sd
    assert "spectral_shift_score" in sd
    assert sd["signal_degradation_state"] in {"NOMINAL", "EMERGING", "ELEVATED"}
    assert "signal_degradation" in latest.get("explanations", {})
