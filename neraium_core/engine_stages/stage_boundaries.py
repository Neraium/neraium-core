from __future__ import annotations

from typing import TypedDict


class EngineStageBoundary(TypedDict):
    stage: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    risk: str
    extraction_candidate: bool


def structural_engine_stage_groups() -> list[EngineStageBoundary]:
    return [
        {
            "stage": "ingress_and_history_buffers",
            "inputs": ("frame", "sensor_schema", "history_ring"),
            "outputs": ("vector", "history_matrix", "history_timestamps"),
            "risk": "low",
            "extraction_candidate": True,
        },
        {
            "stage": "warmup_and_default_payload",
            "inputs": ("frame",),
            "outputs": ("result_payload",),
            "risk": "low",
            "extraction_candidate": True,
        },
        {
            "stage": "representation_and_data_quality",
            "inputs": ("history_matrix", "history_timestamps"),
            "outputs": ("baseline_window", "recent_window", "data_quality", "temporal_features"),
            "risk": "medium",
            "extraction_candidate": True,
        },
        {
            "stage": "multivariate_scoring_and_regime_updates",
            "inputs": ("baseline_window", "recent_window", "regime_store"),
            "outputs": ("drift_scores", "graph_metrics", "regime_memory"),
            "risk": "high",
            "extraction_candidate": False,
        },
        {
            "stage": "decision_projection_and_explanation",
            "inputs": ("scores", "causal_analysis", "history_state"),
            "outputs": ("decision", "risk_assessment", "explanation", "product_flags"),
            "risk": "high",
            "extraction_candidate": False,
        },
    ]
