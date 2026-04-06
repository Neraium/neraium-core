import numpy as np

from fd00x.config import DetectorConfig
from fd00x.detector import StructuralDriftDetector
from fd00x.sii import SII


def test_qit_threshold_mapping_uses_qit_threshold_fields() -> None:
    cfg = DetectorConfig(
        green_yellow=0.60,
        yellow_red=0.80,
        qit_elevated=0.38,
        qit_caution=0.52,
        qit_critical=0.72,
    )
    detector = StructuralDriftDetector(cfg)
    qit_cfg = detector._to_qit_config()

    assert qit_cfg.amber_enter == 0.38
    assert qit_cfg.yellow_enter == 0.52
    assert qit_cfg.red_enter == 0.72


def test_sii_state_mapping_uses_caution_not_warning() -> None:
    baseline = np.zeros((32, 3), dtype=float)
    model = SII(["s0", "s1", "s2"], baseline)

    assert model._to_state(0.60) == "caution"
