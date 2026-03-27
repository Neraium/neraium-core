from __future__ import annotations

"""Deprecated compatibility shim.

Canonical causal logic lives in ``neraium_core.causal``.
This passthrough remains only for temporary compatibility and will be removed in a future release.
"""
DEPRECATED: This module exists only for temporary compatibility.
All new code must import from `neraium_core.causal`.
This file will be removed in a future version.
"""

import warnings

from .causal import causal_metrics, granger_causality_matrix

# TODO(legacy-cleanup): remove this compatibility shim after downstream imports
# migrate from `neraium_core.casual` -> `neraium_core.causal`.
warnings.warn(
    "neraium_core.casual is deprecated; use neraium_core.causal",
    DeprecationWarning,
    stacklevel=2,
)
from .causal import causal_metrics, granger_causality_matrix

__all__ = ["causal_metrics", "granger_causality_matrix"]
