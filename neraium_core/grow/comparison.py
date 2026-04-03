from __future__ import annotations

from statistics import median
from typing import Any


def compare_entities(anchor: dict[str, Any], peers: list[dict[str, Any]]) -> dict[str, Any]:
    if not peers:
        return {
            "baseline_delta": 0.0,
            "peer_median_delta": 0.0,
            "best_performer_delta": 0.0,
            "ranking_position": 1,
            "peer_count": 1,
            "relative_drift": 0.0,
            "relative_inefficiency": 0.0,
            "likely_causal_differences": [],
        }

    peer_drifts = [float(peer.get("structural_drift_score", 0.0)) for peer in peers]
    peer_ineff = [
        float(peer.get("energy_inefficiency", 0.0))
        + float(peer.get("climate_inefficiency", 0.0))
        + float(peer.get("process_inefficiency", 0.0))
        + float(peer.get("biological_inefficiency", 0.0))
        for peer in peers
    ]
    anchor_ineff = (
        float(anchor.get("energy_inefficiency", 0.0))
        + float(anchor.get("climate_inefficiency", 0.0))
        + float(anchor.get("process_inefficiency", 0.0))
        + float(anchor.get("biological_inefficiency", 0.0))
    )
    sorted_by_risk = sorted(peers, key=lambda peer: float(peer.get("composite_instability", 0.0)), reverse=True)
    rank = next((index + 1 for index, peer in enumerate(sorted_by_risk) if peer.get("scope_id") == anchor.get("scope_id")), 1)
    best = min(peer_ineff) if peer_ineff else 0.0
    peer_median = median(peer_ineff) if peer_ineff else 0.0
    causal_differences = []
    peer_drivers = {peer.get("cross_system_driver") for peer in peers if peer.get("cross_system_driver")}
    if anchor.get("cross_system_driver") and anchor.get("cross_system_driver") not in peer_drivers:
        causal_differences.append(f"Anchor driver differs from peer set: {anchor.get('cross_system_driver')}")
    if anchor.get("dominant_relationship_shift"):
        causal_differences.append(str(anchor["dominant_relationship_shift"]))
    return {
        "baseline_delta": round(float(anchor.get("structural_drift_score", 0.0)), 3),
        "peer_median_delta": round(float(anchor.get("structural_drift_score", 0.0)) - median(peer_drifts), 3) if peer_drifts else 0.0,
        "best_performer_delta": round(anchor_ineff - best, 3),
        "ranking_position": rank,
        "peer_count": len(peers),
        "relative_drift": round(float(anchor.get("structural_drift_score", 0.0)) - (median(peer_drifts) if peer_drifts else 0.0), 3),
        "relative_inefficiency": round(anchor_ineff - peer_median, 3),
        "likely_causal_differences": causal_differences[:4],
    }
