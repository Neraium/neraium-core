# Neraium Core Boundary (Frozen Product Surface)

Neraium Core is the **bounded decision-grade layer** that may influence production operator output.

## What ships (production/core)

1. Latent structural state (`structural_state/latent_state.py`)
2. Transition dynamics (`transition_model/transition_dynamics.py`)
3. Trajectory intelligence (`trajectory_memory`, `forecast/trajectory_conditioned.py`)
4. Intervention intelligence + intervention memory (`intervention_intelligence`, `intervention_memory`)
5. Reliability and calibration (`reliability/*`)
6. Real-world validation loop (`validation/*` + `tools/run_validation.py`)
7. Bounded operator-facing output (`platform.py`, `adapters/compatibility.py`, `orchestration/production.py`)

## Advisory (informative, non-production-driving)

- mechanism discovery
- structural law candidates
- law decision support
- cross-system intelligence

These can be shown to operators/reviewers but do not directly drive the production compatibility recommendation.

## Experimental (research only)

- universal layer
- falsification analysis
- active learning
- structural sandbox and what-if simulation

These are excluded from production recommendation flow.

## Enforced behavior

- `ProductionIntelligenceOrchestrator` contains only decision-grade production layers.
- `StructuralSystemIntelligencePlatform` computes compatibility output from production intelligence only.
- Advisory/experimental sections remain available for inspection, but are isolated from production output.

## Evidence and governance principles

- Core philosophy: **not impressively intelligent, reliably correct and safely uncertain**.
- High novelty + weak support must degrade to monitor/bounded advisory with explicit human-review posture.
- Structural Uncertainty Mode is part of production-safe output and compatibility output for downstream gating.
- Intervention memory must materially influence ranking when support and context match are strong.
- Intervention ranking includes bounded correlation-trap penalties (generic-improvement and stabilizing-context checks), not causal claims.
- Outcome attribution remains conservative and includes explicit confidence/assumption fields.
- Reliability calibration is bounded and penalizes repeated high-confidence misses.
- Law governance requires real-world and intervention-sensitive evidence for decision-grade stages.
