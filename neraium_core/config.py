from dataclasses import dataclass


@dataclass(slots=True)
class EngineConfig:
    baseline_window: int = 24
    recent_window: int = 8
    max_frames: int = 500
    mahal_weight: float = 0.65
    cov_weight: float = 0.35
    smoothing_window: int = 3
    enable_vector_smoothing: bool = True
