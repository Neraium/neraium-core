from neraium_core.structural_upgrade import build_evidence_block


def test_build_evidence_block_accepts_string_top_driver():
    out = build_evidence_block(
        frame_count=12,
        temporal_quality={"coverage_ratio": 0.9, "temporal_consistency_score": 0.8},
        stability={"overall_structural_stability": 0.7},
        attribution={"top_drivers": ["s3", "s8"]},
        explanation_count=2,
    )

    ref = out["attribution_reference"]
    assert ref["driver_count"] == 2
    assert ref["top_driver"] == "s3"
