from __future__ import annotations

from typing import Any


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _normalize(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total <= 1e-9:
        return [1.0 / max(1, len(weights)) for _ in weights]
    return [w / total for w in weights]


def track_competing_hypotheses(
    *,
    trajectory: dict[str, Any],
    mechanism: dict[str, Any],
    laws: dict[str, Any],
    transition: dict[str, Any],
) -> dict[str, Any]:
    family = str(trajectory.get("current_trajectory_path_family", transition.get("transition_path", "unknown")))
    support = float(trajectory.get("support_count", 0.0))
    novelty = float(trajectory.get("novelty_score", 0.5))
    contradiction = float((laws.get("falsification_summary") or {}).get("contradiction_rate", 0.0))

    primary_mech = str((((mechanism.get("mechanism_candidates") or [{}])[0]).get("mechanism", "top_relationship")))
    law_items = laws.get("law_candidates") or []
    top_law = str((law_items[0] or {}).get("law", "structural coupling drift")) if law_items else "structural coupling drift"

    raw = [
        {
            "hypothesis": f"Drift-dominant degradation propagated through {primary_mech}",
            "support": _clip01(0.55 + 0.35 * (1.0 if family in {"drift", "escalating"} else 0.0) + 0.1 * min(1.0, support / 6.0)),
            "contradiction_rate": _clip01(0.6 * contradiction + (0.2 if family == "oscillatory" else 0.05)),
        },
        {
            "hypothesis": "Oscillatory instability with phase-coupled amplification",
            "support": _clip01(0.45 + 0.4 * (1.0 if family in {"oscillatory", "phase_shift"} else 0.0) + 0.15 * novelty),
            "contradiction_rate": _clip01(0.5 * contradiction + (0.25 if family in {"drift", "escalating"} else 0.05)),
        },
        {
            "hypothesis": f"Localized failure predicted by law candidate: {top_law}",
            "support": _clip01(0.35 + 0.3 * min(1.0, support / 8.0) + 0.2 * novelty),
            "contradiction_rate": _clip01(0.55 * contradiction + 0.15),
        },
    ]

    weights = [_clip01(r["support"] * (1.0 - 0.7 * r["contradiction_rate"])) for r in raw]
    likelihoods = _normalize(weights)

    active_hypotheses = []
    for row, likelihood in zip(raw, likelihoods):
        uncertainty = _clip01(0.5 * (1.0 - likelihood) + 0.3 * novelty + 0.2 * row["contradiction_rate"])
        active_hypotheses.append(
            {
                "hypothesis": row["hypothesis"],
                "likelihood": round(float(likelihood), 4),
                "support": round(float(row["support"]), 4),
                "contradiction_rate": round(float(row["contradiction_rate"]), 4),
                "uncertainty": round(float(uncertainty), 4),
            }
        )

    active_hypotheses.sort(key=lambda h: h["likelihood"], reverse=True)
    return {
        "active_hypotheses": active_hypotheses,
        "hypothesis_confidence_distribution": [
            {"hypothesis": h["hypothesis"], "confidence": h["likelihood"]} for h in active_hypotheses
        ],
        "status": "experimental",
        "advisory": True,
    }
