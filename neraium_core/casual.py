from __future__ import annotations

"""Deprecated compatibility shim.

Canonical causal logic lives in ``neraium_core.causal``.
This passthrough remains only for temporary compatibility and will be removed in a future release.
"""

import warnings

from .causal import causal_metrics, granger_causality_matrix

warnings.warn(
    "neraium_core.casual is deprecated; use neraium_core.causal",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["causal_metrics", "granger_causality_matrix"]
