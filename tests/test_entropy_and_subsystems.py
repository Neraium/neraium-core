from __future__ import annotations

import numpy as np

from neraium_core.entropy import interaction_entropy
from neraium_core.subsystems import discover_subsystems, subsystem_spectral_measures


def test_entropy_handles_zero_mass() -> None:
    assert interaction_entropy(np.zeros((2, 2))) == 0.0


def test_subsystems_handle_single_sensor() -> None:
    corr = np.array([[1.0]])
    assert discover_subsystems(corr) == []
    measures = subsystem_spectral_measures(corr)
    assert measures["subsystem_instability"] == 0.0
