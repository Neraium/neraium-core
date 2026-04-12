"""Configuration constants for the structural detection engine.

These values are locked for production compatibility (Phase-1 FD004 policy).
Changing these constants requires careful regression testing.
"""

# Baseline adaptation: how quickly rolling baseline responds to new data.
# Locked at 0.92 to avoid absorbing instability signals.
DEFAULT_BASELINE_ADAPTATION_ALPHA = 0.92

# Composite score must be below this AND system in nominal state for baseline update.
BASELINE_UPDATE_MAX_COMPOSITE = 0.85

# Number of recent interpreted states used to compute classification stability.
CLASSIFICATION_STABILITY_WINDOW = 15

# Transition memory window size for historical tracking.
TRANSITION_MEMORY_WINDOW = 8

# Transition pressure thresholds for emerging/sustained classification.
TRANSITION_EMERGING_THRESHOLD = 0.85
TRANSITION_SUSTAINED_THRESHOLD = 1.15

# Fast mode geometry computation interval and downsampling.
FAST_MODE_GEOMETRY_UPDATE_INTERVAL = 3
FAST_MODE_GEOMETRY_DOWNSAMPLE_STEP = 2

# Minimum baseline samples collected before alert thresholds can be calibrated.
# Prevents early single-sample spikes from triggering alerts.
MIN_BASELINE_SAMPLES_FOR_CALIBRATION = 28

# FD004 policy defaults (locked for Phase-1 compatibility).
# These drive the drift-score state machine behavior.
LOCKED_FD004_POLICY_DEFAULTS = {
    "drift_smoothing_window": 25,
    "watch_quantile": 0.65,
    "alert_quantile": 0.85,
    "watch_persistence": 5,
    "alert_persistence": 3,
    "fast_trigger_multiplier": 1.25,
    "alert_latch_enabled": True,
    "unlatch_ratio": 0.75,
}

# Unpack FD004 defaults for backward compatibility.
DEFAULT_DRIFT_SMOOTHING_WINDOW = LOCKED_FD004_POLICY_DEFAULTS["drift_smoothing_window"]
DEFAULT_WATCH_QUANTILE = LOCKED_FD004_POLICY_DEFAULTS["watch_quantile"]
DEFAULT_ALERT_QUANTILE = LOCKED_FD004_POLICY_DEFAULTS["alert_quantile"]
DEFAULT_WATCH_PERSISTENCE = LOCKED_FD004_POLICY_DEFAULTS["watch_persistence"]
DEFAULT_ALERT_PERSISTENCE = LOCKED_FD004_POLICY_DEFAULTS["alert_persistence"]
DEFAULT_FAST_TRIGGER_MULTIPLIER = LOCKED_FD004_POLICY_DEFAULTS["fast_trigger_multiplier"]
DEFAULT_ALERT_LATCH_ENABLED = LOCKED_FD004_POLICY_DEFAULTS["alert_latch_enabled"]
DEFAULT_UNLATCH_RATIO = LOCKED_FD004_POLICY_DEFAULTS["unlatch_ratio"]

# Sensor value normalization.
DEFAULT_SENSOR_NORMALIZATION_ALPHA = 0.01

# Drift EMA smoothing coefficient.
DEFAULT_DRIFT_EMA_ALPHA = 0.2

# Default transition persistence length.
DEFAULT_TRANSITION_PERSISTENCE = 3
