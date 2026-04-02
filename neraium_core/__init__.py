"""Core package for Neraium."""

from importlib.metadata import PackageNotFoundError, version

from neraium_core.sports_betting_adapter import build_betting_frame
from neraium_core.stock_market_adapter import build_stock_frame
from neraium_core.trading_signals import map_structural_output_to_signal


def load_market_data(*args, **kwargs):
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
