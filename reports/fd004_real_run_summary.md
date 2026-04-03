## Scope note
- This report describes the current read-only Neraium analytics layer.
- Results are decision-support outputs for human operators; no automated control or actuation is performed.

# FD004 Real Pipeline Run Summary

## Execution
- Initial run with requested `data/fd004/...` paths failed because those paths were not present in this repo layout.
- Full root-file run started but was long-running in this environment; fallback subset run was used (first 5 units, first 100 cycles each).
- Subset command succeeded and generated outputs under `fd004_outputs_subset/`.

## Output Files
- `fd004_outputs_subset/fd004_real_report.json`
- `fd004_outputs_subset/fd004_real_timeseries.csv`
- `fd004_outputs_subset/fd004_real_rul_map.json`

## Summary Stats (from report JSON)
- `units_total`: `5`
- `rows_processed`: `500`
- `train_path`: `data/fd004_subset/train_FD004.txt`
- `test_path`: `data/fd004_subset/test_FD004.txt`
- `rul_path`: `data/fd004_subset/RUL_FD004.txt`
- units reached MEDIUM: `5`
- units reached HIGH: `5`
- MEDIUM before HIGH (among units reaching both): `5/5`
- `average_early_warning_window`: `1.0` cycles

## Phase Classification
- `LOW` → `stable`
- `MEDIUM` → `drift`
- `HIGH` → `unstable`
- UNKNOWN labels are no longer emitted in the FD004 real evaluation timeseries.

## Per-unit timing metrics

| unit | first_MEDIUM_step | first_HIGH_step | early_warning_window |
|---|---:|---:|---:|
| unit_001 | 25 | 26 | 1 |
| unit_002 | 25 | 26 | 1 |
| unit_003 | 25 | 26 | 1 |
| unit_004 | 25 | 26 | 1 |
| unit_005 | 25 | 26 | 1 |

## Instability/Drift vs RUL observations
- Instability-to-RUL correlation is negative for all subset units (`-0.642` to `-0.8826`), indicating instability rises as estimated RUL decreases.
- Drift-to-RUL correlation is also negative (`-0.1965` to `-0.4829`), but generally weaker than instability.
- Threshold-window checks (`<100`, `<50`, `<30` estimated RUL) show instability increases are most consistently present before the `<50` threshold in this subset.

## RUL threshold increase checks by unit

| unit | <100 | <50 | <30 |
|---|---|---|---|
| unit_001 | false | true | true |
| unit_002 | true | true | false |
| unit_003 | false | false | false |
| unit_004 | false | false | false |
| unit_005 | false | false | false |

## Sanity Checks
- LOW→HIGH immediate jumps: `0` of `495` transitions (ratio `0.0000`).
- Units with positive instability slope over time: `5/5`.

## First 10 rows of `fd004_real_timeseries.csv`

```csv
asset_id,cycle,structural_drift_score,composite_instability,trend,phase,risk_level,estimated_rul,operator_message,structural_analysis_available
unit_001,1,0.0,0.0,stable,stable,LOW,121,System appears stable based on current heuristic interpretation.,False
unit_001,2,0.0,0.0,stable,stable,LOW,120,System appears stable based on current heuristic interpretation.,False
unit_001,3,0.0,0.0,stable,stable,LOW,119,System appears stable based on current heuristic interpretation.,False
unit_001,4,0.0,0.0,stable,stable,LOW,118,System appears stable based on current heuristic interpretation.,False
unit_001,5,0.0,0.0,stable,stable,LOW,117,System appears stable based on current heuristic interpretation.,False
unit_001,6,0.0,0.0,stable,stable,LOW,116,System appears stable based on current heuristic interpretation.,False
unit_001,7,0.0,0.0,stable,stable,LOW,115,System appears stable based on current heuristic interpretation.,False
unit_001,8,0.0,0.0,stable,stable,LOW,114,System appears stable based on current heuristic interpretation.,False
unit_001,9,0.0,0.0,stable,stable,LOW,113,System appears stable based on current heuristic interpretation.,False
unit_001,10,0.0,0.0,stable,stable,LOW,112,System appears stable based on current heuristic interpretation.,False
```

## RUL map sample
- `unit_001`: `22`
- `unit_002`: `39`
- `unit_003`: `107`
- `unit_004`: `75`
- `unit_005`: `149`
