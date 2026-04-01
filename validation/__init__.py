"""Real-world validation pipeline for Neraium."""

<<<<<<< HEAD
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import RealWorldValidationPipeline

__all__ = ["RealWorldValidationPipeline"]


def __getattr__(name: str):
    if name == "RealWorldValidationPipeline":
        from .pipeline import RealWorldValidationPipeline

        return RealWorldValidationPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
=======
from .pipeline import RealWorldValidationPipeline

__all__ = ["RealWorldValidationPipeline"]
>>>>>>> 3aedc7e4a4597802300889285e2fecafec7b2eee
