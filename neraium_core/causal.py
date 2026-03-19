from __future__ import annotations

# Compatibility wrapper: the codebase contains `casual.py` (typo) but other
# modules import `causal.py`. Re-export the same functions to keep runtime stable.
from .casual import causal_metrics, granger_causality_matrix

__all__ = ["causal_metrics", "granger_causality_matrix"]

