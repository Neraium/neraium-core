from __future__ import annotations

# Compatibility wrapper: the repository keeps `regime_store.py` at the project
# root, but some modules import it as `neraium_core.regime_store`.
from regime_store import RegimeStore

__all__ = ["RegimeStore"]

