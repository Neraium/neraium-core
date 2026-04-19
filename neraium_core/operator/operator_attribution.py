"""
operator_attribution.py — Structural attribution in operator space.

Attribution answers the question: given that the structural operator has
drifted (||A_t - A_0||_F > 0), which directed couplings are responsible,
and what does the pattern of change imply?

This replaces the removed generate_hypotheses / generate_validation_plan
machinery in causal.py.  Unlike those functions, attribution here is
grounded in the actual operator change ΔA = A_t - A_0 rather than in
linear combinations of scalar heuristics.

Key concepts:
    Row attribution:    Row i of ΔA tells you how the predictors of signal i
                        have changed.  ||ΔA[i, :]|| is the total incoming
                        coupling change for signal i.

    Column attribution: Column j of ΔA tells you how signal j's influence
                        over all other signals has changed.
                        ||ΔA[:, j]|| is the total outgoing coupling change.

    Coupling gain:      ΔA[i, j] > 0 means signal j now drives signal i more
                        strongly than at baseline.
    Coupling loss:      ΔA[i, j] < 0 means signal j now drives signal i less
                        than at baseline.

    Structural pattern:
        - If a small number of columns dominate ||ΔA||, the change is
          source-localised: one (or few) signals changed how they drive
          the rest of the system.
        - If a small number of rows dominate, the change is
          target-localised: one (or few) signals became harder to predict.
        - If change is diffuse across both rows and columns, the change
          is systemic (global coupling restructuring).

    Stability contribution:
        The spectral radius of A_t is upper-bounded by ||A_t||_F.  Rows
        (or columns) with large ||ΔA[i,:]|| contributed most to the shift
        in the spectral radius.  More precisely, by the Frobenius perturbation
        bound: |ρ(A_t) - ρ(A_0)| ≤ ||ΔA||_F, so per-element |ΔA[i,j]|
        gives a conservative bound on each coupling's contribution to
        destabilisation.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .structural_operator import StructuralOperator


def attribute_operator_drift(
    current: StructuralOperator,
    baseline: StructuralOperator,
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Decompose structural drift ΔA = A_t - A_0 into signal-level attributions.

    Args:
        current:  Current structural operator 𝒮_t.
        baseline: Baseline structural operator 𝒮_0.
        top_k:    Number of top coupling changes to return.

    Returns:
        Dict with:
            operator_drift:         ||A_t - A_0||_F  (scalar)
            innovation_drift:       ||Σ_t - Σ_0||_F  (scalar)
            top_coupling_changes:   List of top-k (source, target, delta) triples
            incoming_attribution:   Per-signal incoming coupling change ||ΔA[i,:]||
            outgoing_attribution:   Per-signal outgoing coupling change ||ΔA[:,j]||
            localisation_pattern:   "source" | "target" | "systemic" | "diffuse"
            stability_shift:        ρ(A_t) - ρ(A_0)  (positive = destabilising)
    """
    if current.n_signals != baseline.n_signals:
        raise ValueError(
            f"Signal count mismatch: current={current.n_signals}, baseline={baseline.n_signals}. "
            "Cannot attribute drift across operators of different dimensionality."
        )

    drift = current.drift_from(baseline)
    delta_A = current.coupling_change(baseline)
    N = current.n_signals
    names: list[str] = (
        list(current.signal_names) if current.signal_names else [str(i) for i in range(N)]
    )

    # Per-signal incoming (row) and outgoing (column) attribution
    incoming = np.linalg.norm(delta_A, axis=1)   # shape (N,)
    outgoing = np.linalg.norm(delta_A, axis=0)   # shape (N,)

    incoming_total = float(incoming.sum()) or 1.0
    outgoing_total = float(outgoing.sum()) or 1.0

    incoming_attr = [
        {"signal": names[i], "signal_idx": i, "change_norm": round(float(incoming[i]), 6),
         "share": round(float(incoming[i] / incoming_total), 4)}
        for i in np.argsort(incoming)[::-1]
    ]
    outgoing_attr = [
        {"signal": names[j], "signal_idx": j, "change_norm": round(float(outgoing[j]), 6),
         "share": round(float(outgoing[j] / outgoing_total), 4)}
        for j in np.argsort(outgoing)[::-1]
    ]

    # Localisation pattern: compare concentration (entropy) of incoming vs outgoing
    def _concentration(v: np.ndarray) -> float:
        """Herfindahl-style concentration: 1 = fully concentrated, 1/N = fully diffuse."""
        total = float(v.sum())
        if total <= 0 or N == 0:
            return 0.0
        p = v / total
        return float((p ** 2).sum())

    conc_in = _concentration(incoming)
    conc_out = _concentration(outgoing)
    # Threshold: concentration above 1/N + 0.2 suggests localisation
    diffuse_threshold = 1.0 / max(N, 1) + 0.2
    if conc_out > diffuse_threshold and conc_out >= conc_in:
        pattern = "source"       # few signals changed their outgoing influence
    elif conc_in > diffuse_threshold and conc_in > conc_out:
        pattern = "target"       # few signals became harder to predict
    elif conc_in > diffuse_threshold and conc_out > diffuse_threshold:
        pattern = "source"       # both concentrated — report dominant
    else:
        pattern = "systemic"     # diffuse change across many couplings

    stability_shift = round(current.spectral_radius - baseline.spectral_radius, 6)

    return {
        "operator_drift": round(drift["operator_drift"], 6),
        "innovation_drift": round(drift["innovation_drift"], 6),
        "stability_shift": stability_shift,
        "localisation_pattern": pattern,
        "top_coupling_changes": current.top_coupling_changes(baseline, k=top_k),
        "incoming_attribution": incoming_attr,
        "outgoing_attribution": outgoing_attr,
        "concentration": {
            "incoming": round(conc_in, 4),
            "outgoing": round(conc_out, 4),
            "diffuse_threshold": round(diffuse_threshold, 4),
        },
        "note": (
            "Attribution is grounded in ΔA = A_t - A_0 (operator-space coupling change). "
            "Incoming attribution ranks signals by change in how well they are predicted. "
            "Outgoing attribution ranks signals by change in how much they drive others."
        ),
    }


def summarise_structural_change(
    current: StructuralOperator,
    baseline: StructuralOperator,
) -> str:
    """
    Return a one-paragraph plain-language summary of the structural change.

    This is the correct replacement for the removed causal.py hypothesis text:
    grounded in actual operator change, not in heuristic interpolation.
    """
    attr = attribute_operator_drift(current, baseline, top_k=3)
    dA = attr["operator_drift"]
    pattern = attr["localisation_pattern"]
    ss = attr["stability_shift"]
    top = attr["top_coupling_changes"]

    stability_text = (
        f"Stability margin {'decreased' if ss > 0 else 'increased'} "
        f"by {abs(ss):.4f} (spectral radius now {current.spectral_radius:.4f})."
    )

    if top:
        top_pair = top[0]
        link_text = (
            f"The largest single coupling change is "
            f"{top_pair['source_name']} → {top_pair['target_name']} "
            f"({'strengthened' if top_pair['delta'] > 0 else 'weakened'} "
            f"by {abs(top_pair['delta']):.4f})."
        )
    else:
        link_text = "No dominant coupling pair identified."

    pattern_text = {
        "source": "Change is source-localised: one or few signals altered their outgoing influence.",
        "target": "Change is target-localised: one or few signals became harder to predict.",
        "systemic": "Change is systemic: coupling restructuring is diffuse across the operator.",
        "diffuse": "Change is diffuse with no dominant structural signature.",
    }.get(pattern, "Pattern undetermined.")

    return (
        f"Operator drift ||dA||_F = {dA:.4f}. "
        f"{stability_text} "
        f"{pattern_text} "
        f"{link_text}"
    )
