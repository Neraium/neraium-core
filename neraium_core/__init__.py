"""Core package for Neraium."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("neraium-core")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["alignment", "doctrine", "models", "pipeline", "service", "store", "sii", "__version__"]
