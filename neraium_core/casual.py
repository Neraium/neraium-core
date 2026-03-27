from __future__ import annotations

"""
DEPRECATED COMPATIBILITY SHIM.

This module is deprecated. Canonical causal runtime logic now lives in
`neraium_core.causal`, and all new imports must use `neraium_core.causal`.

This shim remains only for temporary backward compatibility and will be
removed in a future version.
"""

import warnings

from .causal import causal_metrics, granger_causality_matrix

warnings.warn(
    "neraium_core.casual is deprecated; import from neraium_core.causal instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["causal_metrics", "granger_causality_matrix"]
