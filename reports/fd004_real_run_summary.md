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

## Sanity Checks
- LOW→HIGH immediate jumps: `0` of `495` transitions (ratio `0.0000`).
- Units with positive instability slope over time: `5/5`.

## First 10 rows of `fd004_real_timeseries.csv`

```
asset_id,cycle,structural_drift_score,composite_instability,trend,risk_level,operator_message,structural_analysis_available
unit_001,1,0.0,0.0,UNKNOWN,LOW,System appears stable based on current heuristic interpretation.,False
unit_001,2,0.0,0.0,UNKNOWN,LOW,System appears stable based on current heuristic interpretation.,False
unit_001,3,0.0,0.0,UNKNOWN,LOW,System appears stable based on current heuristic interpretation.,False
unit_001,4,0.0,0.0,UNKNOWN,LOW,System appears stable based on current heuristic interpretation.,False
unit_001,5,0.0,0.0,UNKNOWN,LOW,System appears stable based on current heuristic interpretation.,False
unit_001,6,0.0,0.0,UNKNOWN,LOW,System appears stable based on current heuristic interpretation.,False
unit_001,7,0.0,0.0,UNKNOWN,LOW,System appears stable based on current heuristic interpretation.,False
unit_001,8,0.0,0.0,UNKNOWN,LOW,System appears stable based on current heuristic interpretation.,False
unit_001,9,0.0,0.0,UNKNOWN,LOW,System appears stable based on current heuristic interpretation.,False
unit_001,10,0.0,0.0,UNKNOWN,LOW,System appears stable based on current heuristic interpretation.,False
```

## RUL map sample
- `unit_001`: `22`
- `unit_002`: `39`
- `unit_003`: `107`
- `unit_004`: `75`
- `unit_005`: `149`
