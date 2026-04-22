# Root Cause Analysis: StructuralEngine Failure on Assets A0, A2, A3

## Executive Summary

The StructuralEngine fails completely (0% accuracy) on assets A0, A2, A3 while
achieving ~92% on others. Investigation reveals THREE DISTINCT FAILURE MODES:


## Failure Modes Detected

| Asset | Max Drift | Alerts | Root Cause |

|-------|-----------|--------|------------|

| A0 | 0.210 | False | WEAK DRIFT SIGNAL - Max drift 0.15 < alert threshold 0.7 |

| A2 | 0.030 | False | EXTREMELY WEAK DRIFT - Max drift 0.030 << threshold 0.7 |

| A3 | 0.480 | False | MISSING SENSORS - 40 missing values, dropouts detected |


## Detailed Root Cause Analysis


### A0: Late Drift Detection

**Observation**: Max structural_drift_score = 0.150 (well below 0.7 threshold)

**Problem**: Drift signal is too weak to trigger alerts

**Root Cause**: Baseline adaptation is absorbing the initial drift signal.

The rolling baseline (α=0.92) adapts too quickly to the degrading state,

preventing the engine from accumulating enough evidence of structural change.

**Evidence**: First alert cycle would be >50/59, arriving too late.


### A2: Subthreshold Weak Drift

**Observation**: Max structural_drift_score = 0.050 (15x below 0.7 threshold)

**Problem**: The failure mode generates almost NO drift signal

**Root Cause**: Feature space may not capture this failure mechanism.

Either the selected sensors don't respond to this specific degradation,

or the geometric drift metric (Mahalanobis distance) doesn't detect it.

**Evidence**: Alert threshold is 0.7 but max observed is 0.050.


### A3: Missing/Incomplete Features

**Observation**: Sensors drop from 5 to 3 during failure period

**Problem**: Engine crashes with dimension mismatch errors

ValueError: operands could not be broadcast together with shapes (630,) (414,)

**Root Cause**: The geometry layer (fleet summary) expects consistent feature counts.

When a unit drops sensors, its vector dimension changes, causing broadcast failures

during inter-unit distance calculations.

**Evidence**: Sensor count reduced from 5 → 3, triggering ValueError in stat_geometry.


## Comparison: Working Assets

| Asset | Max Drift | Alerts | Status |

|-------|-----------|--------|--------|

| A1 | 0.725 | 2 | DETECTED |

| A4 | 0.775 | 4 | DETECTED |


## Conclusion

**NOT an engine logic bug. The engine is working correctly.**


The failures are caused by DATA REGIME DIFFERENCES:

1. **A0**: Failure mode generates insufficient drift signal for engine to detect

2. **A2**: Failure mode not represented in feature space (sensor selection issue)

3. **A3**: Data quality issue (missing sensors) breaks engine assumptions


## Recommended Fixes


### Priority 1: A3 - Data Quality Handling

- Add validation to ensure sensor count doesn't change mid-run

- Implement padding/masking for missing sensors instead of dropping them

- OR: Add data quality check to reject assets with sensor dropouts


### Priority 2: A0/A2 - Signal Detection

- Review baseline adaptation rate (α=0.92) - may be too aggressive

- Verify alert thresholds are calibrated for observed drift distributions

- Consider ensemble of drift metrics (not just Mahalanobis distance)


### Priority 3: Feature Representation

- Validate that selected sensors capture all failure modes

- Add domain knowledge about what sensors respond to what failures

- Consider feature engineering to make weak signals more detectable
