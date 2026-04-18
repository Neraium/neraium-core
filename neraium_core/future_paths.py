"""Future Path Map: System Navigation Engine

Infers plausible future trajectories from current structural state and recommends
interventions to navigate toward stable recovery paths.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class CurrentState:
    """Snapshot of system's current structural health."""
    regime_name: Optional[str]
    health_band: str  # "stable" | "watch" | "critical"
    confidence: float  # 0.0-1.0
    summary: str
    structural_drift: float
    relational_stability: float
    temporal_consistency: float


@dataclass
class EtaRange:
    """Time-to-event estimate in cycles."""
    min: int
    max: int
    unit: str = "cycles"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FuturePath:
    """A plausible future trajectory of the system."""
    path_id: str
    label: str
    probability: float  # 0.0-1.0
    risk: str  # "low" | "medium" | "high"
    eta_range: EtaRange
    trend_summary: str
    drivers: List[str]
    point_of_no_return: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["eta_range"] = self.eta_range.to_dict()
        return d


@dataclass
class Intervention:
    """Recommended action to shift system state."""
    action: str
    expected_effect: str
    target_paths: List[str]  # path_ids this intervention affects
    confidence: float  # 0.0-1.0


@dataclass
class FuturePathMap:
    """Complete future paths analysis."""
    current_state: CurrentState
    future_paths: List[FuturePath]
    recommended_interventions: List[Intervention]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_state": asdict(self.current_state),
            "future_paths": [p.to_dict() for p in self.future_paths],
            "recommended_interventions": [asdict(i) for i in self.recommended_interventions],
        }


class FuturePathMapper:
    """Infers plausible future paths from current system state."""

    def __init__(self):
        self.DRIFT_ACCELERATION_WINDOW = 8
        self.STABILITY_TREND_WINDOW = 8
        self.TRAJECTORY_CONVERGENCE_THRESHOLD = 0.15

    @staticmethod
    def from_engine(engine: Any, current_frame: Dict[str, Any]) -> FuturePathMap:
        """Build future path map from StructuralEngine state.

        Args:
            engine: StructuralEngine instance with analysis history
            current_frame: Current frame from engine.process_frame()

        Returns:
            FuturePathMap with current state, paths, and interventions
        """
        mapper = FuturePathMapper()
        return mapper.analyze(engine, current_frame)

    def analyze(self, engine: Any, current_frame: Dict[str, Any]) -> FuturePathMap:
        """Analyze current state and infer future trajectories.

        Pipeline:
        1. Extract current state from engine outputs
        2. Compute trajectory metrics (drift slope, stability trend)
        3. Score path probabilities using heuristic rules
        4. Generate intervention recommendations
        """
        # === Current State ===
        current_state = self._extract_current_state(engine, current_frame)

        # === Path Inference ===
        future_paths = self._infer_future_paths(engine, current_state)

        # === Interventions ===
        interventions = self._generate_interventions(current_state, future_paths, engine)

        return FuturePathMap(
            current_state=current_state,
            future_paths=future_paths,
            recommended_interventions=interventions,
        )

    def _extract_current_state(self, engine: Any, frame: Dict[str, Any]) -> CurrentState:
        """Extract and summarize current structural state."""
        # Get structural metrics from latest engine outputs
        latest = engine.latest_result or {}

        # Use structural_drift_score if available, otherwise score
        structural_drift_raw = float(latest.get("structural_drift_score", latest.get("score", 0.0)))
        # Normalize drift score: if > 1.0, assume it's in raw scale and cap at 1.0
        # The StructuralEngine uses scores that can exceed 1.0
        drift_score = min(1.0, structural_drift_raw / 100.0) if structural_drift_raw > 1.0 else float(structural_drift_raw)
        interpreted_state = str(latest.get("interpreted_state", "UNKNOWN")).upper()

        # Extract relational stability from causal analysis if available
        causal_analysis = latest.get("causal_analysis", {})
        relational_stability = float(causal_analysis.get("stability", 0.5))

        # Temporal consistency from historical analysis
        temporal_consistency = self._compute_temporal_consistency(engine)

        # Map interpreted state to health band
        health_band = self._state_to_health_band(interpreted_state, drift_score)

        # Get regime name
        regime_name = latest.get("regime_name")

        # Build confidence from history stability
        confidence = self._compute_confidence(engine, interpreted_state)

        # Summary
        summary = self._build_current_state_summary(
            interpreted_state, drift_score, relational_stability, health_band
        )

        return CurrentState(
            regime_name=regime_name,
            health_band=health_band,
            confidence=confidence,
            summary=summary,
            structural_drift=drift_score,
            relational_stability=relational_stability,
            temporal_consistency=temporal_consistency,
        )

    def _infer_future_paths(
        self, engine: Any, current_state: CurrentState
    ) -> List[FuturePath]:
        """Score and rank plausible future paths.

        Paths:
        1. Recovery: drift decreases, stability improves
        2. Degradation: persistent drift, partial decoupling
        3. Failure: accelerating drift, stability collapse
        """
        # Compute trajectory metrics
        drift_slope, drift_accel = self._compute_drift_dynamics(engine)
        stability_slope = self._compute_stability_trend(engine)
        transition_activity = self._compute_transition_activity(engine)

        # Score each path
        recovery_score = self._score_recovery_path(
            drift_slope, drift_accel, stability_slope, current_state
        )
        degradation_score = self._score_degradation_path(
            drift_slope, drift_accel, stability_slope, current_state
        )
        failure_score = self._score_failure_path(
            drift_slope, drift_accel, stability_slope, transition_activity, current_state
        )

        # Normalize probabilities
        scores = [recovery_score, degradation_score, failure_score]
        total = sum(scores)
        if total < 0.01:
            probs = [1.0 / 3.0] * 3
        else:
            probs = [s / total for s in scores]

        paths = [
            FuturePath(
                path_id="recovery",
                label="Stable Recovery",
                probability=probs[0],
                risk="low",
                eta_range=EtaRange(
                    min=self._estimate_eta_min(recovery_score, "recovery"),
                    max=self._estimate_eta_max(recovery_score, "recovery"),
                ),
                trend_summary="System likely re-stabilizes if current structure holds",
                drivers=[
                    "improving relational stability" if stability_slope > 0 else "stable relationships",
                    "lower drift acceleration" if drift_accel < 0 else "moderate drift",
                ],
                point_of_no_return=None,
            ),
            FuturePath(
                path_id="degradation",
                label="Efficiency Loss",
                probability=probs[1],
                risk="medium",
                eta_range=EtaRange(
                    min=self._estimate_eta_min(degradation_score, "degradation"),
                    max=self._estimate_eta_max(degradation_score, "degradation"),
                ),
                trend_summary="System likely enters degraded but operating condition",
                drivers=[
                    "persistent drift" if drift_slope > 0.05 else "steady state",
                    "partial decoupling of sensor relationships",
                ],
                point_of_no_return=self._estimate_point_of_no_return(
                    current_state.structural_drift, "degradation"
                ),
            ),
            FuturePath(
                path_id="failure",
                label="Instability / Failure",
                probability=probs[2],
                risk="high",
                eta_range=EtaRange(
                    min=self._estimate_eta_min(failure_score, "failure"),
                    max=self._estimate_eta_max(failure_score, "failure"),
                ),
                trend_summary="System is approaching a structurally unstable regime",
                drivers=[
                    "accelerating drift" if drift_accel > 0.1 else "rising drift",
                    "loss of relational stability",
                ],
                point_of_no_return=self._estimate_point_of_no_return(
                    current_state.structural_drift, "failure"
                ),
            ),
        ]

        return sorted(paths, key=lambda p: p.probability, reverse=True)

    def _score_recovery_path(
        self,
        drift_slope: float,
        drift_accel: float,
        stability_slope: float,
        current_state: CurrentState,
    ) -> float:
        """Recovery path: low drift, improving stability, low acceleration."""
        score = 0.0

        # Drift decreasing or flat = strong recovery signal
        if drift_slope < -0.05:
            score += 0.6
        elif drift_slope < 0.0:
            score += 0.5
        elif drift_slope < 0.05:
            score += 0.3

        # Stability improving = recovery signal
        if stability_slope > 0.05:
            score += 0.4
        elif stability_slope > 0.0:
            score += 0.3

        # Negative acceleration (deceleration) = recovery signal
        if drift_accel < -0.05:
            score += 0.2
        elif drift_accel < 0.0:
            score += 0.1

        # Current health band influences recovery likelihood (but not if already critical)
        if current_state.health_band == "stable":
            score += 0.4
        elif current_state.health_band == "watch":
            score += 0.2
        # critical band does not add recovery bonus

        # High drift actively penalizes recovery score
        if current_state.structural_drift > 0.7:
            score *= 0.5

        return max(0.0, score)

    def _score_degradation_path(
        self,
        drift_slope: float,
        drift_accel: float,
        stability_slope: float,
        current_state: CurrentState,
    ) -> float:
        """Degradation path: moderate drift, weakening stability."""
        score = 0.0

        # Positive drift slope but moderate = degradation
        if 0.05 < drift_slope < 0.2:
            score += 0.4
        elif drift_slope >= 0.05:
            score += 0.2

        # Stability flat or slightly declining = degradation
        if -0.1 < stability_slope <= 0.0:
            score += 0.3
        elif stability_slope <= -0.1:
            score += 0.15

        # Moderate acceleration = degradation trajectory
        if 0.0 < drift_accel < 0.1:
            score += 0.25

        # Current health band
        if current_state.health_band == "watch":
            score += 0.35
        elif current_state.health_band == "stable":
            score += 0.15

        return max(0.0, score)

    def _score_failure_path(
        self,
        drift_slope: float,
        drift_accel: float,
        stability_slope: float,
        transition_activity: float,
        current_state: CurrentState,
    ) -> float:
        """Failure path: rising drift, collapsing stability, high transitions."""
        score = 0.0

        # High drift slope = strong failure trajectory
        if drift_slope > 0.1:
            score += 0.5
        elif drift_slope > 0.05:
            score += 0.35
        elif drift_slope > 0.0:
            score += 0.15

        # Stability declining sharply = failure signal
        if stability_slope < -0.1:
            score += 0.35
        elif stability_slope < -0.05:
            score += 0.25
        elif stability_slope < 0.0:
            score += 0.1

        # Positive acceleration = failure signal
        if drift_accel > 0.05:
            score += 0.35
        elif drift_accel > 0.0:
            score += 0.2

        # High transition activity = system instability
        if transition_activity > 0.6:
            score += 0.25

        # Current health band
        if current_state.health_band == "critical":
            score += 0.4
        elif current_state.health_band == "watch":
            score += 0.2

        return max(0.0, score)

    def _generate_interventions(
        self,
        current_state: CurrentState,
        paths: List[FuturePath],
        engine: Any,
    ) -> List[Intervention]:
        """Generate actionable interventions based on risk and path analysis."""
        interventions = []

        # Identify highest-risk path
        failure_path = next((p for p in paths if p.path_id == "failure"), None)
        degradation_path = next((p for p in paths if p.path_id == "degradation"), None)

        # If significant risk of failure, recommend multiple interventions
        if failure_path and failure_path.probability > 0.15:
            interventions.append(
                Intervention(
                    action="Reduce operating load",
                    expected_effect="Decreases probability of failure path by easing structural stress",
                    target_paths=["failure", "degradation"],
                    confidence=0.65,
                )
            )

            interventions.append(
                Intervention(
                    action="Inspect subsystem with strongest structural divergence",
                    expected_effect="Identifies and corrects root cause of instability",
                    target_paths=["failure"],
                    confidence=0.55,
                )
            )

        # If moderate degradation risk
        if degradation_path and degradation_path.probability > 0.35:
            interventions.append(
                Intervention(
                    action="Rebalance operation",
                    expected_effect="Redistributes load to improve relational stability",
                    target_paths=["degradation"],
                    confidence=0.58,
                )
            )

        # If drift is rising
        if current_state.structural_drift > 0.5:
            interventions.append(
                Intervention(
                    action="Shorten maintenance interval",
                    expected_effect="Catches drift earlier, reduces acceleration",
                    target_paths=["failure", "degradation"],
                    confidence=0.52,
                )
            )

        # If confidence in current state is low
        if current_state.confidence < 0.4:
            interventions.append(
                Intervention(
                    action="Increase sensor coverage or calibration verification",
                    expected_effect="Improves visibility and reduces uncertainty",
                    target_paths=["recovery", "degradation", "failure"],
                    confidence=0.48,
                )
            )

        return interventions

    def _compute_drift_dynamics(self, engine: Any) -> tuple[float, float]:
        """Compute slope and acceleration of structural drift.

        Returns:
            (slope, acceleration) of drift over recent window
        """
        drift_history = list(engine.score_history) if hasattr(engine, "score_history") else []

        if len(drift_history) < 2:
            return 0.0, 0.0

        # Use last DRIFT_ACCELERATION_WINDOW samples
        window = drift_history[-self.DRIFT_ACCELERATION_WINDOW :]

        # Slope: linear regression of last samples
        if len(window) >= 2:
            x = np.arange(len(window), dtype=float)
            y = np.asarray(window, dtype=float)
            try:
                coeffs = np.polyfit(x, y, 1)
                slope = float(coeffs[0])
            except (ValueError, TypeError):
                slope = 0.0
        else:
            slope = 0.0

        # Acceleration: second derivative
        if len(window) >= 3:
            try:
                coeffs = np.polyfit(x, y, 2)
                accel = float(coeffs[0]) * 2.0  # Second derivative of quadratic
            except (ValueError, TypeError):
                accel = 0.0
        else:
            accel = 0.0

        return slope, accel

    def _compute_stability_trend(self, engine: Any) -> float:
        """Compute trend of relational stability over recent frames."""
        stability_history = []

        # Extract stability scores from recent frame results
        if hasattr(engine, "frames") and engine.frames:
            for frame in list(engine.frames)[-self.STABILITY_TREND_WINDOW :]:
                if "_stability_score" in frame:
                    stability_history.append(float(frame.get("_stability_score", 0.5)))

        if len(stability_history) < 2:
            return 0.0

        # Slope of stability over time
        x = np.arange(len(stability_history), dtype=float)
        y = np.asarray(stability_history, dtype=float)
        try:
            coeffs = np.polyfit(x, y, 1)
            return float(coeffs[0])
        except (ValueError, TypeError):
            return 0.0

    def _compute_transition_activity(self, engine: Any) -> float:
        """Compute how active recent state transitions have been."""
        if hasattr(engine, "_transition_pressure_history"):
            history = list(engine._transition_pressure_history)
            if history:
                return float(np.mean(history))
        return 0.0

    def _compute_temporal_consistency(self, engine: Any) -> float:
        """Measure consistency of temporal patterns in recent data."""
        if hasattr(engine, "geometry_layer"):
            # Proxy: use spectral consistency
            return 0.7  # Placeholder; would compute from actual time-series stats
        return 0.5

    def _compute_confidence(self, engine: Any, interpreted_state: str) -> float:
        """Estimate confidence in current state classification."""
        # Base confidence from state history stability
        if hasattr(engine, "_state_history"):
            state_history = list(engine._state_history)
            if len(state_history) > 5:
                # If all recent states are the same, high confidence
                recent = state_history[-5:]
                if all(s == recent[0] for s in recent):
                    return 0.85
                # If mostly consistent
                if sum(1 for s in recent if s == recent[0]) >= 4:
                    return 0.70
                # Otherwise moderate
                return 0.50
        return 0.60

    def _state_to_health_band(self, interpreted_state: str, drift_score: float) -> str:
        """Map interpreted state and drift score to health band."""
        if drift_score > 0.7:
            return "critical"
        elif drift_score > 0.4 or interpreted_state == "WATCH":
            return "watch"
        else:
            return "stable"

    def _build_current_state_summary(
        self, state: str, drift: float, stability: float, health_band: str
    ) -> str:
        """Build human-readable summary of current state."""
        if health_band == "critical":
            return f"System in critical condition: drift {drift:.2f}, stability {stability:.2f}. Immediate attention required."
        elif health_band == "watch":
            return f"System under observation: drift rising to {drift:.2f}. Stability at {stability:.2f}."
        else:
            return f"System stable: drift {drift:.2f}, stable relationships. Operating normally."

    def _estimate_eta_min(self, path_score: float, path_id: str) -> int:
        """Estimate minimum cycles to critical state."""
        # Higher score = more confident about path = sooner arrival
        if path_id == "recovery":
            # Recovery takes time: 20-80 cycles typical
            return max(20, int(100 * (1.0 - path_score)))
        elif path_id == "degradation":
            # Degradation: 10-40 cycles
            return max(10, int(50 * (1.0 - path_score)))
        else:  # failure
            # Failure: 5-18 cycles (fastest)
            return max(5, int(20 * (1.0 - path_score)))

    def _estimate_eta_max(self, path_score: float, path_id: str) -> int:
        """Estimate maximum cycles to critical state."""
        if path_id == "recovery":
            return min(80, int(150 * (1.0 - path_score * 0.5)))
        elif path_id == "degradation":
            return min(40, int(75 * (1.0 - path_score * 0.5)))
        else:  # failure
            return min(18, int(30 * (1.0 - path_score * 0.5)))

    def _estimate_point_of_no_return(self, drift_score: float, path_id: str) -> Optional[int]:
        """Estimate cycles until point of no return (irreversible state)."""
        if path_id == "degradation":
            # For degradation, point of no return is around 70% drift
            return int(10 + (0.7 - drift_score) * 20) if drift_score < 0.7 else 5
        elif path_id == "failure":
            # For failure, point of no return is around 80% drift
            return int(8 + (0.8 - drift_score) * 15) if drift_score < 0.8 else 3
        return None
