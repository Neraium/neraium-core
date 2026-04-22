"""Auxiliary analytics: optional enrichment layer.

These modules provide advanced insights but are NOT required for
core structural detection. They can be disabled via feature flags
without breaking the core runtime.

Modules:
- causal_intelligence: Causal analysis, hypotheses, ranked actions
- graph_analysis: Causal graphs, propagation, root cause chains
- hierarchy_analysis: Cascade direction, localization scoring
- counterfactual_guidance: Reversibility classification, futures
- path_prototypes: Trajectory shape features, directional evolution
- horizon_analysis: Risk horizon estimation
- constraint_analysis: Lock-in and commitment scoring
- branching_analysis: Branching decision paths

Core runtime (engine/) must NOT depend on auxiliary modules.
Access is always gated through orchestration layer.
"""

__all__ = []
