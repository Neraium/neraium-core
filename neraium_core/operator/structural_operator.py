"""
structural_operator.py — VAR(1) structural operator estimation.

This module implements the central mathematical object of Neraium:

    𝒮_t = (A_t, Σ_t)

estimated from a rolling window of z-normalized observations via Yule-Walker
equations. Every other structural signal in the system is derived from or
approximated by this object.

Theory:
    Given T observations X ∈ ℝ^{T×N}, the VAR(1) model is:

        X(t) = A · X(t-1) + ε(t),   ε(t) ~ N(0, Σ)

    The Yule-Walker estimator solves this by computing two moment matrices:

        Γ_0 = (1/T) Xᵀ · X           [contemporaneous covariance]
        Γ_1 = (1/(T-1)) X[:-1]ᵀ · X[1:]  [lagged cross-covariance]

    Then:
        A   = Γ_1 · (Γ_0 + λI)^{-1}   [regularized to handle ill-conditioning]
        Σ   = Γ_0 - A · Γ_0 · Aᵀ       [innovation covariance]

    For z-normalized inputs (unit marginal variance), Γ_0 ≈ R_t (the
    correlation matrix computed by geometry.py) and Γ_1 ≈ C_t (the lagged
    correlation matrix computed by directional.py). The assembly
    A = C_t · R_t^{-1} is what no file in the repo currently computes,
    despite all the required pieces being present.

Key structural properties derived from (A, Σ):
    Spectral radius:   ρ(A) = max|λ_i(A)|
        ρ < 1.0   → stable dynamics (attenuating)
        ρ = 1.0   → marginally stable (unit circle)
        ρ > 1.0   → unstable dynamics (amplifying; pre-failure region)
    Stability margin:  m = 1 - ρ(A)  (positive = stable; approaching 0 = critical)
    Operator drift:    d_A = ||A_t - A_0||_F
    Innovation drift:  d_Σ = ||Σ_t - Σ_0||_F

Regularization note:
    The ridge parameter λ prevents inversion failure when Γ_0 is ill-conditioned
    (typical when N is large relative to T). Default λ=1e-4 is conservative;
    increase for short windows or high-dimensional systems.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


ArrayLike = Any

# Default Tikhonov regularization for (Γ_0 + λI)^{-1}
_DEFAULT_LAMBDA: float = 1e-4


@dataclass(frozen=True)
class StructuralOperator:
    """
    The VAR(1) structural operator 𝒮 = (A, Σ) for a multivariate system.

    Attributes:
        A:               Propagation matrix ∈ ℝ^{N×N}.  A[i,j] encodes
                         how strongly signal j at time t-1 predicts signal i at time t.
        Sigma:           Innovation covariance ∈ ℝ^{N×N}.  Σ[i,j] encodes
                         the residual co-movement not explained by A.
        n_signals:       Number of signals N.
        n_obs:           Number of observations T used in estimation.
        regularization:  λ used in (Γ_0 + λI)^{-1}.
        signal_names:    Optional list of signal names for attribution readability.
    """

    A: np.ndarray
    Sigma: np.ndarray
    n_signals: int
    n_obs: int
    regularization: float = _DEFAULT_LAMBDA
    signal_names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.A.shape != (self.n_signals, self.n_signals):
            raise ValueError(f"A must be ({self.n_signals}, {self.n_signals}); got {self.A.shape}")
        if self.Sigma.shape != (self.n_signals, self.n_signals):
            raise ValueError(f"Sigma must be ({self.n_signals}, {self.n_signals}); got {self.Sigma.shape}")

    # ------------------------------------------------------------------
    # Stability properties (derived from A)
    # ------------------------------------------------------------------

    @property
    def spectral_radius(self) -> float:
        """ρ(A) = max|λ_i(A)|.  Values > 1 indicate instability."""
        if self.n_signals == 0:
            return 0.0
        eigenvalues = np.linalg.eigvals(self.A)
        return float(np.max(np.abs(eigenvalues)))

    @property
    def stability_margin(self) -> float:
        """m = 1 - ρ(A).  Positive = stable; approaching 0 = critical; negative = unstable."""
        return 1.0 - self.spectral_radius

    @property
    def is_stable(self) -> bool:
        """True iff ρ(A) < 1 (all eigenvalues strictly inside the unit circle)."""
        return self.spectral_radius < 1.0

    @property
    def dominant_mode(self) -> np.ndarray:
        """Right eigenvector corresponding to the eigenvalue of largest absolute magnitude."""
        if self.n_signals == 0:
            return np.array([], dtype=float)
        eigenvalues, eigenvectors = np.linalg.eig(self.A)
        idx = int(np.argmax(np.abs(eigenvalues)))
        return np.real(eigenvectors[:, idx])

    @property
    def coupling_asymmetry(self) -> float:
        """||A - Aᵀ||_F / (||A||_F + 1e-12).  Fraction of coupling that is directed."""
        fro_a = float(np.linalg.norm(self.A, ord="fro"))
        if fro_a < 1e-12:
            return 0.0
        return float(np.linalg.norm(self.A - self.A.T, ord="fro")) / fro_a

    @property
    def innovation_entropy(self) -> float:
        """Shannon entropy of |Σ| as a distribution.  Rising entropy → dissolving constraints."""
        flat = np.abs(self.Sigma).ravel()
        total = flat.sum()
        if total <= 0:
            return 0.0
        p = flat / total
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    # ------------------------------------------------------------------
    # Structural distance to another operator
    # ------------------------------------------------------------------

    def drift_from(self, baseline: "StructuralOperator") -> dict[str, float]:
        """
        Compute structural drift between self and a baseline operator.

        Returns:
            operator_drift:    ||A_t - A_0||_F
            innovation_drift:  ||Σ_t - Σ_0||_F
            total_drift:       sqrt(operator_drift² + innovation_drift²)
        """
        if self.A.shape != baseline.A.shape:
            raise ValueError(
                f"Operator shape mismatch: {self.A.shape} vs {baseline.A.shape}.  "
                "Cannot compute drift between operators of different dimensionality."
            )
        dA = float(np.linalg.norm(self.A - baseline.A, ord="fro"))
        dS = float(np.linalg.norm(self.Sigma - baseline.Sigma, ord="fro"))
        return {
            "operator_drift": dA,
            "innovation_drift": dS,
            "total_drift": float(np.sqrt(dA**2 + dS**2)),
        }

    # ------------------------------------------------------------------
    # Innovation residuals
    # ------------------------------------------------------------------

    def innovation_residuals(self, observations: ArrayLike) -> np.ndarray:
        """
        Compute innovation residuals ε(t) = X(t) - A · X(t-1).

        These residuals are the correct input for critical slowing down (CSD)
        analysis (variance, lag-1 autocorrelation).  Applying CSD to raw
        observations X(t) conflates structural change with level change;
        applying it to ε(t) isolates the structural signal.

        Args:
            observations: Array of shape (T, N) — the same window used to
                          estimate this operator or a subsequent window.
        Returns:
            residuals: Array of shape (T-1, N).
        """
        X = np.asarray(observations, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_signals:
            raise ValueError(
                f"Observations must be (T, {self.n_signals}); got {X.shape}"
            )
        if X.shape[0] < 2:
            return np.empty((0, self.n_signals), dtype=float)
        past = np.nan_to_num(X[:-1], nan=0.0)
        present = np.nan_to_num(X[1:], nan=0.0)
        predicted = past @ self.A.T
        return present - predicted

    # ------------------------------------------------------------------
    # Attribution
    # ------------------------------------------------------------------

    def coupling_change(self, baseline: "StructuralOperator") -> np.ndarray:
        """
        Element-wise coupling change matrix ΔA = A_t - A_0.

        ΔA[i, j] > 0 means signal j has gained influence over signal i since baseline.
        ΔA[i, j] < 0 means signal j has lost influence over signal i since baseline.

        Returns:
            delta_A: (N, N) array.
        """
        if self.A.shape != baseline.A.shape:
            raise ValueError("Shape mismatch in coupling_change")
        return self.A - baseline.A

    def top_coupling_changes(
        self,
        baseline: "StructuralOperator",
        k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Return the top-k (source, target) signal pairs by absolute coupling change.

        This is the correct structural attribution: which directed links in the
        propagation graph have changed most from baseline, and in which direction?

        Args:
            baseline: The baseline operator 𝒮_0.
            k:        Number of top pairs to return.
        Returns:
            List of dicts with keys: source_idx, target_idx, source_name, target_name,
            delta, direction ('strengthened' | 'weakened').
        """
        delta = self.coupling_change(baseline)
        N = self.n_signals
        names: list[str] = list(self.signal_names) if self.signal_names else [str(i) for i in range(N)]

        flat_idx = np.argsort(np.abs(delta).ravel())[::-1]
        results: list[dict[str, Any]] = []
        for fi in flat_idx[:k]:
            i, j = int(fi // N), int(fi % N)
            d = float(delta[i, j])
            results.append({
                "source_idx": j,
                "target_idx": i,
                "source_name": names[j],
                "target_name": names[i],
                "delta": round(d, 6),
                "direction": "strengthened" if d > 0 else "weakened",
            })
        return results

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "A": self.A.tolist(),
            "Sigma": self.Sigma.tolist(),
            "n_signals": self.n_signals,
            "n_obs": self.n_obs,
            "regularization": self.regularization,
            "signal_names": list(self.signal_names),
            "spectral_radius": round(self.spectral_radius, 6),
            "stability_margin": round(self.stability_margin, 6),
            "is_stable": self.is_stable,
            "coupling_asymmetry": round(self.coupling_asymmetry, 6),
            "innovation_entropy": round(self.innovation_entropy, 6),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StructuralOperator":
        return cls(
            A=np.asarray(d["A"], dtype=float),
            Sigma=np.asarray(d["Sigma"], dtype=float),
            n_signals=int(d["n_signals"]),
            n_obs=int(d["n_obs"]),
            regularization=float(d.get("regularization", _DEFAULT_LAMBDA)),
            signal_names=tuple(str(s) for s in d.get("signal_names", [])),
        )


def fit_structural_operator(
    observations: ArrayLike,
    *,
    regularization: float = _DEFAULT_LAMBDA,
    signal_names: list[str] | None = None,
) -> StructuralOperator:
    """
    Estimate the VAR(1) structural operator 𝒮 = (A, Σ) from an observation window.

    Uses Yule-Walker equations.  Inputs are expected to be z-normalized
    (zero mean, unit variance per signal) to keep Γ_0 ≈ R_t numerically.
    For raw (un-normalized) inputs, the estimator still works but Σ will
    be in raw signal units.

    Algorithm:
        1. Center columns (demean).
        2. Γ_0 = (1/T) Xᵀ X        [contemporaneous sample covariance]
        3. Γ_1 = (1/(T-1)) X[:-1]ᵀ X[1:]  [lagged sample cross-covariance]
        4. A   = Γ_1 · (Γ_0 + λI)^{-1}
        5. Σ   = Γ_0 - A · Γ_0 · Aᵀ
        6. Symmetrize Σ and clip to positive-semidefinite.

    Args:
        observations:  (T, N) array.  Requires T ≥ N+2 for reliable estimation;
                       works for T < N but increases regularization dependence.
        regularization: λ in (Γ_0 + λI)^{-1}.  Increase if Γ_0 is ill-conditioned.
        signal_names:  Optional list of length N for readable attribution output.

    Returns:
        StructuralOperator with fields A, Sigma, and derived stability properties.

    Raises:
        ValueError: if observations are not 2D or have fewer than 2 rows.
    """
    X = np.asarray(observations, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Observations must be 2D (T, N); got shape {X.shape}")
    T, N = X.shape
    if T < 2:
        raise ValueError(f"At least 2 observations required; got T={T}")
    if N == 0:
        A = np.empty((0, 0), dtype=float)
        Sigma = np.empty((0, 0), dtype=float)
        return StructuralOperator(A=A, Sigma=Sigma, n_signals=0, n_obs=T)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 1 — demean
    mu = X.mean(axis=0)
    Xc = X - mu

    # Step 2 — contemporaneous covariance Γ_0
    gamma0 = (Xc.T @ Xc) / T
    np.fill_diagonal(gamma0, np.diag(gamma0))  # ensure exactly symmetric

    # Step 3 — lagged cross-covariance Γ_1
    gamma1 = (Xc[:-1].T @ Xc[1:]) / max(T - 1, 1)

    # Step 4 — regularized Yule-Walker: A = Γ_1 · (Γ_0 + λI)^{-1}
    lam = max(float(regularization), 0.0)
    gamma0_reg = gamma0 + lam * np.eye(N, dtype=float)
    try:
        A = gamma1 @ np.linalg.inv(gamma0_reg)
    except np.linalg.LinAlgError:
        # Should not happen with λ > 0; fallback to pseudo-inverse
        A = gamma1 @ np.linalg.pinv(gamma0_reg)

    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 5 — innovation covariance Σ = Γ_0 - A · Γ_0 · Aᵀ
    Sigma = gamma0 - A @ gamma0 @ A.T
    # Step 6 — symmetrize and project to PSD
    Sigma = (Sigma + Sigma.T) / 2.0
    # Clip negative eigenvalues (numerical noise from finite sample)
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 0.0)
    Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
    Sigma = np.nan_to_num(Sigma, nan=0.0, posinf=0.0, neginf=0.0)

    names: tuple[str, ...] = ()
    if signal_names is not None:
        names = tuple(str(s) for s in signal_names[:N])

    return StructuralOperator(
        A=A,
        Sigma=Sigma,
        n_signals=N,
        n_obs=T,
        regularization=float(regularization),
        signal_names=names,
    )


def operator_drift(
    current: StructuralOperator,
    baseline: StructuralOperator,
    *,
    include_innovation: bool = True,
) -> dict[str, float]:
    """
    Compute structural drift between current and baseline operators.

    This is the primary structural health signal.  Unlike the correlation
    Frobenius drift ||R_t - R_0||_F, operator drift captures directed
    coupling change and is sensitive to dynamics that do not appear in the
    stationary covariance.

    Args:
        current:            The current structural operator 𝒮_t.
        baseline:           The baseline structural operator 𝒮_0.
        include_innovation: Whether to include ||Σ_t - Σ_0||_F.

    Returns:
        Dict with keys:
            operator_drift     — ||A_t - A_0||_F
            innovation_drift   — ||Σ_t - Σ_0||_F  (if include_innovation)
            total_drift        — joint Frobenius distance in (A, Σ) space
    """
    return current.drift_from(baseline)


def structural_stability(op: StructuralOperator) -> dict[str, float | bool | list[float]]:
    """
    Compute stability properties of a structural operator.

    The spectral radius ρ(A) is the key stability indicator:
        ρ < 1:  stable regime    (dynamics decay toward steady state)
        ρ = 1:  marginal regime  (dynamics neither decay nor amplify)
        ρ > 1:  unstable regime  (dynamics amplify; pre-failure or failure)

    The stability margin m = 1 - ρ quantifies distance to the unit circle.
    Monitoring m over time gives a continuous early warning signal.

    Args:
        op: A StructuralOperator instance.

    Returns:
        Dict with:
            spectral_radius:    ρ(A)
            stability_margin:   1 - ρ(A)
            is_stable:          ρ(A) < 1.0
            dominant_mode:      right eigenvector of dominant eigenvalue
            coupling_asymmetry: fraction of directed (asymmetric) coupling
            innovation_entropy: entropy of Σ (constraint tightness indicator)
    """
    return {
        "spectral_radius": round(op.spectral_radius, 6),
        "stability_margin": round(op.stability_margin, 6),
        "is_stable": op.is_stable,
        "dominant_mode": [round(float(v), 6) for v in op.dominant_mode],
        "coupling_asymmetry": round(op.coupling_asymmetry, 6),
        "innovation_entropy": round(op.innovation_entropy, 6),
    }
