"""
SII-ML v2.0.

Adds learnable sensor coupling and graph-aware representations on top of the
atomic SII score stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from .settings import get_optimal_config
from .sii import SII, ScoreProgressionConfig


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_shift = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x_shift)
    return ex / np.clip(ex.sum(axis=axis, keepdims=True), 1e-12, None)


@dataclass
class MLConfig:
    attention_heads: int = 4
    attention_head_dim: int = 16
    graph_hidden: int = 32
    neural_boost: float = 0.25
    online_lr: float = 0.0005


class SensorCouplingAttention:
    """Small multi-head self-attention over sensor features."""

    def __init__(self, num_sensors: int, heads: int = 4, head_dim: int = 16, seed: int = 7) -> None:
        self.num_sensors = num_sensors
        self.heads = heads
        self.head_dim = head_dim
        rng = np.random.default_rng(seed)
        self.w_q = rng.normal(0.0, 0.12, (heads, 1, head_dim))
        self.w_k = rng.normal(0.0, 0.12, (heads, 1, head_dim))
        self.w_v = rng.normal(0.0, 0.12, (heads, 1, head_dim))

    def forward(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z_in = z.reshape(-1, 1)
        contexts = []
        attention_mats = []
        for h in range(self.heads):
            q = z_in @ self.w_q[h]
            k = z_in @ self.w_k[h]
            v = z_in @ self.w_v[h]
            scores = (q @ k.T) / max(np.sqrt(self.head_dim), 1.0)
            attn = _softmax(scores, axis=1)
            context = attn @ v
            contexts.append(context)
            attention_mats.append(attn)
        stacked_context = np.concatenate(contexts, axis=1)
        mean_attn = np.mean(np.stack(attention_mats, axis=0), axis=0)
        sensor_importance = mean_attn.mean(axis=0)
        return stacked_context, sensor_importance


class AdaptiveGraphLearner:
    """Graph learner with simple GCN-style propagation."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, seed: int = 11) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.08, (input_dim, hidden_dim))
        self.w2 = rng.normal(0.0, 0.08, (hidden_dim, hidden_dim))

    def forward(self, context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sim = context @ context.T
        sim = np.abs(sim)
        adj = sim / np.clip(np.max(sim), 1e-8, None)
        deg = np.diag(np.sum(adj, axis=1) + 1e-8)
        deg_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(deg)))
        norm_adj = deg_inv_sqrt @ adj @ deg_inv_sqrt
        h1 = np.tanh(norm_adj @ context @ self.w1)
        h2 = np.tanh(norm_adj @ h1 @ self.w2)
        centrality = np.sum(norm_adj, axis=1)
        centrality = centrality / np.clip(centrality.sum(), 1e-12, None)
        return h2, centrality


class NeuralBooster:
    """Tiny MLP that predicts a correction scalar from 5 fundamental scores."""

    def __init__(self, seed: int = 17) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.2, (5, 64))
        self.b1 = np.zeros(64)
        self.w2 = rng.normal(0.0, 0.16, (64, 32))
        self.b2 = np.zeros(32)
        self.w3 = rng.normal(0.0, 0.12, (32, 16))
        self.b3 = np.zeros(16)
        self.w4 = rng.normal(0.0, 0.1, (16, 1))
        self.b4 = np.zeros(1)

    def forward(self, x: np.ndarray) -> float:
        h1 = np.maximum(0.0, x @ self.w1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.w2 + self.b2)
        h3 = np.maximum(0.0, h2 @ self.w3 + self.b3)
        out = 1.0 / (1.0 + np.exp(-(h3 @ self.w4 + self.b4)))
        return float(out[0])


class SIIML(SII):
    """SII with attention + graph learning + neural score fusion."""

    def __init__(
        self,
        sensors: Iterable[str],
        baseline_data: np.ndarray,
        *,
        preset: str = "balanced",
        config: Mapping[str, object] | None = None,
    ) -> None:
        merged = get_optimal_config(preset)
        if config:
            for key, value in config.items():
                if isinstance(value, dict) and key in merged:
                    merged[key].update(value)
                else:
                    merged[key] = value

        thresholds = merged.get("thresholds", {})
        progression_cfg = merged.get("progression", {})
        progression = ScoreProgressionConfig(
            baseline_fraction=float(progression_cfg.get("baseline_fraction", 0.20)),
            drift_weight=float(progression_cfg.get("drift_weight", 0.30)),
            variance_weight=float(progression_cfg.get("variance_weight", 0.22)),
            alpha=float(progression_cfg.get("alpha", 3.0)),
            raw_offset=float(progression_cfg.get("raw_offset", 1.10)),
        )
        super().__init__(sensors, baseline_data, thresholds=thresholds, progression=progression)
        ml_cfg = merged["ml"]
        self.ml_config = MLConfig(
            attention_heads=int(ml_cfg["attention_heads"]),
            attention_head_dim=int(ml_cfg["attention_head_dim"]),
            graph_hidden=int(ml_cfg["graph_hidden"]),
            neural_boost=float(ml_cfg["neural_boost"]),
            online_lr=float(ml_cfg["online_lr"]),
        )
        self.attention = SensorCouplingAttention(
            num_sensors=len(self.sensors),
            heads=self.ml_config.attention_heads,
            head_dim=self.ml_config.attention_head_dim,
        )
        self.graph = AdaptiveGraphLearner(
            input_dim=self.ml_config.attention_heads * self.ml_config.attention_head_dim,
            hidden_dim=self.ml_config.graph_hidden,
        )
        self.booster = NeuralBooster()

    def assess(
        self,
        readings: Mapping[str, float],
        *,
        current_cycle_index: int | None = None,
        total_cycles: int | None = None,
    ) -> Dict[str, object]:
        base = super().assess(
            readings,
            current_cycle_index=current_cycle_index,
            total_cycles=total_cycles,
        )
        z = np.asarray([base["z_scores"][sensor] for sensor in self.sensors], dtype=float)

        context, attention_importance = self.attention.forward(z)
        graph_embed, graph_centrality = self.graph.forward(context)
        _ = graph_embed  # kept for possible downstream diagnostics

        layer_vector = np.asarray(
            [
                base["layer_scores"]["l1"],
                base["layer_scores"]["l2"],
                base["layer_scores"]["l3"],
                base["layer_scores"]["l4"],
                base["layer_scores"]["l5"],
            ],
            dtype=float,
        )
        neural_score = self.booster.forward(layer_vector)
        final_score = (1.0 - self.ml_config.neural_boost) * float(base["score"]) + self.ml_config.neural_boost * neural_score

        fused_importance = 0.5 * np.abs(z) + 0.3 * attention_importance + 0.2 * graph_centrality
        rank_idx = np.argsort(fused_importance)[::-1]
        critical = [self.sensors[i] for i in rank_idx[: min(5, len(rank_idx))]]

        base.update(
            {
                "atomic_score": float(base.get("atomic_score", base["score"])),
                "neural_score": float(neural_score),
                "score": float(np.clip(final_score, 0.0, 1.0)),
                "state": self._to_state(float(final_score)),
                "critical_sensors": critical,
                "sensor_importance": {self.sensors[i]: float(fused_importance[i]) for i in range(len(self.sensors))},
                "attention_importance": {self.sensors[i]: float(attention_importance[i]) for i in range(len(self.sensors))},
                "graph_centrality": {self.sensors[i]: float(graph_centrality[i]) for i in range(len(self.sensors))},
            }
        )
        return base


def create_siiml(
    sensors: Sequence[str],
    baseline_data: np.ndarray,
    *,
    preset: str = "balanced",
    config: Mapping[str, object] | None = None,
) -> SIIML:
    """Factory for SIIML."""

    return SIIML(sensors, baseline_data, preset=preset, config=config)


__all__ = [
    "MLConfig",
    "SensorCouplingAttention",
    "AdaptiveGraphLearner",
    "NeuralBooster",
    "SIIML",
    "create_siiml",
]
