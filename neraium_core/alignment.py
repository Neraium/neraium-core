from collections import deque
import numpy as np

from neraium_core.regime_store import RegimeStore
from neraium_core.causal_graph import causal_graph_metrics
from neraium_core.forecast_models import time_to_threshold_ar1

from neraium_core.geometry import correlation_matrix, structural_drift
from neraium_core.scoring import composite_instability_score
from neraium_core.decision_layer import decision_output


class StructuralEngine:
    def __init__(self):
        self.frames = deque(maxlen=200)
        self.score_history = deque(maxlen=100)

        self.store = RegimeStore()
        data = self.store.load()

        self.regimes = data["regimes"]
        self.baselines = data["baselines"]

    def process_frame(self, frame):
        values = np.array(list(frame["sensor_values"].values()))
        self.frames.append(values)

        if len(self.frames) < 10:
            return {"state": "warming"}

        window = np.array(self.frames)

        corr = correlation_matrix(window)
        drift = structural_drift(corr, corr)

        # --- regime ---
        regime_name = "default"

        if regime_name not in self.baselines:
            self.baselines[regime_name] = corr.tolist()

        regime_corr = np.array(self.baselines[regime_name])
        regime_drift = structural_drift(corr, regime_corr)

        # --- causal ---
        causal = causal_graph_metrics(corr)

        # --- scoring ---
        components = {
            "drift": drift,
            "regime_drift": regime_drift,
            "causal": causal["density"],
        }

        score = composite_instability_score(components)
        self.score_history.append(score)

        forecast = {
            "ar1_time_to_instability": time_to_threshold_ar1(self.score_history)
        }

        decision = decision_output(score, components, forecast)

        # persist
        self.store.save({
            "regimes": self.regimes,
            "baselines": self.baselines
        })

        return {
            **decision,
            "regime_name": regime_name,
            "regime_drift": regime_drift,
            "causal_density": causal["density"],
            "score": score
        }