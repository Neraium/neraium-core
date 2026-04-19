from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

np = pytest.importorskip("numpy")

MODULE_PATH = Path(__file__).resolve().parents[1] / "neraium_core" / "engine_stages" / "regime_mutation.py"
SPEC = importlib.util.spec_from_file_location("regime_mutation_stage", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


RegimeMutationInput = MODULE.RegimeMutationInput
mutate_regime_state = MODULE.mutate_regime_state


class _Context:
    def __init__(self) -> None:
        self.regime_signatures = [{"name": "r0"}]
        self.regime_baselines: dict[str, dict[str, object]] = {}
        self.baseline_locked = False
        self.baseline_adaptation_alpha = 0.92
        self.transition_aware_enabled = True
        self._rolling_baseline_corr = None
        self._stable_manifold_corr = None
        self._baseline_set_at = None
        self._baseline_coverage_samples = 0
        self.persist_calls = 0

    def _persist_regime_state(self) -> None:
        self.persist_calls += 1


def test_regime_mutation_updates_baseline_library_and_transition_state() -> None:
    ctx = _Context()
    corr = np.array([[1.0, 0.2], [0.2, 1.0]], dtype=float)
    result = mutate_regime_state(
        ctx,
        RegimeMutationInput(
            regime_name="r1",
            signature=np.array([0.1, 0.2], dtype=float),
            corr_recent=corr,
            baseline_window=8,
            valid_signal_count=2,
            transition_state="NONE",
            transition_pressure=0.0,
            decision_interpreted_state="NOMINAL_STRUCTURE",
        ),
    )
    assert result.regime_drift == 0.0
    assert ctx.regime_baselines["r1"]["count"] == 1
    assert ctx.persist_calls == 1
    assert result.rolling_baseline_updated is True


def test_regime_mutation_respects_baseline_lock() -> None:
    ctx = _Context()
    ctx.baseline_locked = True
    corr = np.array([[1.0, 0.1], [0.1, 1.0]], dtype=float)
    result = mutate_regime_state(
        ctx,
        RegimeMutationInput(
            regime_name=None,
            signature=np.array([0.0, 0.0], dtype=float),
            corr_recent=corr,
            baseline_window=8,
            valid_signal_count=2,
            transition_state="NONE",
            transition_pressure=0.0,
            decision_interpreted_state="NOMINAL_STRUCTURE",
            apply_regime_library_mutation=False,
            apply_rolling_baseline_mutation=True,
        ),
    )
    assert result.rolling_baseline_updated is False
    assert ctx._rolling_baseline_corr is None
