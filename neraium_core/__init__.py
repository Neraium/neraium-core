"""Core package for Neraium.

Runtime note:
The active deployment/runtime for market operations is the Markets API
(`neraium_core.markets.app.api:app`). Legacy stock-market helpers remain
available only via lazy compatibility wrappers.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from neraium_core.sports_betting_adapter import build_betting_frame


def build_stock_frame(*args: Any, **kwargs: Any):
    """Compatibility wrapper for legacy stock adapter import paths."""
    from neraium_core.stock_market_adapter import build_stock_frame as _build_stock_frame

    return _build_stock_frame(*args, **kwargs)


def map_structural_output_to_signal(*args: Any, **kwargs: Any):
    """Compatibility wrapper for legacy trading signal import paths."""
    from neraium_core.trading_signals import map_structural_output_to_signal as _map_signal

    return _map_signal(*args, **kwargs)


def load_market_data(*args: Any, **kwargs: Any):
    from neraium_core.market_data_loader import load_market_data as _load_market_data

    return _load_market_data(*args, **kwargs)


try:
    __version__ = version("neraium-core")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "alignment",
    "models",
    "pipeline",
    "service",
    "store",
    "sii",
    "build_stock_frame",
    "build_betting_frame",
    "map_structural_output_to_signal",
    "load_market_data",
    "__version__",
]
