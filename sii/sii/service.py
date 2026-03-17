from __future__ import annotations

from datetime import datetime

import pandas as pd

from sii.directional import directional_metrics, lagged_directional_matrix
from sii.early_warning import early_warning_metrics
from sii.entropy import structural_entropy
from sii.forecasting import forecast_instability
from sii.geometry import correlation_geometry, structural_centrality, structural_drift
from sii.graph import build_graph, graph_metrics
from sii.models import (
    DriftMetrics,
    ForecastMetrics,
    GraphMetrics,
    ObservableSnapshot,
    RegimeState,
    SpectralMetrics,
    SubsystemMetrics,
    TelemetryIngestRequest,
    WindowConfig,
    WindowSnapshot,
)
from sii.normalization import zscore_window
from sii.regimes import nearest_regime, window_signature
from sii.scoring import composite_score
from sii.spectral import spectral_observables
from sii.store import SQLiteStore
from sii.subsystems import subsystem_instability
from sii.telemetry import append_frame, points_to_frame
from sii.windows import sliding_windows


class StructuralIntelligenceService:
    def __init__(self, config: WindowConfig | None = None, store: SQLiteStore | None = None) -> None:
        self.config = config or WindowConfig()
        self.store = store or SQLiteStore()
        self.telemetry = pd.DataFrame()
        self.baseline_corr: pd.DataFrame | None = None
        self.regimes: dict[str, object] = {}
        self.score_history: list[float] = []
        self.latest_window: pd.DataFrame | None = None
        self.latest_observables: ObservableSnapshot | None = None
        self.latest_drift: DriftMetrics | None = None
        self.latest_regime: RegimeState | None = None
        self.latest_forecast: ForecastMetrics | None = None

    def ingest(self, payload: TelemetryIngestRequest) -> None:
        incoming = points_to_frame(payload.points)
        self.telemetry = append_frame(self.telemetry, incoming)
        windows = sliding_windows(self.telemetry, self.config.size, self.config.step)
        if not windows:
            return
        self.latest_window = windows[-1]
        self._compute_latest()

    def _compute_latest(self) -> None:
        assert self.latest_window is not None
        z = zscore_window(self.latest_window)
        corr = correlation_geometry(z)
        if self.baseline_corr is None:
            self.baseline_corr = corr.copy()
        fro, mad = structural_drift(self.baseline_corr, corr)
        sig = window_signature(self.latest_window)
        regime_name, regime_dist = nearest_regime(sig, self.regimes)  # type: ignore[arg-type]
        if not self.regimes:
            self.regimes["baseline"] = sig
        _, g = build_graph(corr)
        gm = graph_metrics(g)
        sm = spectral_observables(corr)
        dm = directional_metrics(lagged_directional_matrix(self.latest_window))
        ew = early_warning_metrics(self.latest_window)
        ent = structural_entropy(corr)
        subs = subsystem_instability(corr)
        components = {
            "spectral_radius": sm["spectral_radius"],
            "inverse_spectral_gap": 1.0 / max(sm["spectral_gap"], 1e-6),
            "causal_divergence": dm["causal_divergence"],
            "graph_stability": 1.0 - gm["density"],
            "lag1_autocorr_avg": ew["lag1_autocorr_avg"],
            "baseline_relative_drift": fro,
            "regime_relative_drift": regime_dist,
            "entropy": ent,
            "subsystem_instability": subs["max_subsystem_instability"],
            "forecast_contribution": 0.0,
        }
        score = composite_score(components)
        self.score_history.append(score)
        fcm = forecast_instability(self.score_history)
        components["forecast_contribution"] = max(fcm["recent_slope"], 0.0)
        score = composite_score(components)
        self.score_history[-1] = score

        ts = datetime.utcnow()
        self.latest_drift = DriftMetrics(
            frobenius_drift=fro,
            mean_absolute_drift=mad,
            regime_relative_drift=regime_dist,
        )
        self.latest_regime = RegimeState(regime=regime_name, distance=regime_dist)
        self.latest_forecast = ForecastMetrics(**fcm)
        self.latest_observables = ObservableSnapshot(
            timestamp=ts,
            entropy=ent,
            centrality=structural_centrality(corr),
            graph=GraphMetrics(**gm),
            spectral=SpectralMetrics(
                spectral_radius=sm["spectral_radius"],
                spectral_gap=sm["spectral_gap"],
                dominant_eigenvector=sm["dominant_eigenvector"],
                ranked_signal_loadings=sm["ranked_signal_loadings"],
            ),
            directional=dm,
            early_warning=ew,
            subsystems=SubsystemMetrics(**subs),
            score=score,
        )
        self.store.set("latest_observables", self.latest_observables.model_dump(mode="json"))

    def latest_window_snapshot(self) -> WindowSnapshot | None:
        if self.latest_window is None:
            return None
        return WindowSnapshot(
            start=self.latest_window.index[0],
            end=self.latest_window.index[-1],
            rows=len(self.latest_window),
            columns=list(self.latest_window.columns),
        )
