# Upgraded Multinode SII Benchmark Report

## Run Context
- Seeds: [42, 43, 44, 45, 46]
- Conditions: ['coherent_time', 'disturbed_time']
- Nodes: ['A', 'B', 'C', 'D']
- GAL-2 configured: False
- GAL-2 used for disturbed_time: False

## Aggregate Metrics (mean ± std)

### coherent_time
- alert_precision: 0.813490 ± 0.039730
- alert_recall: 0.945000 ± 0.056319
- baseline_stability_rate: 0.990625 ± 0.000000
- false_positive_rate: 0.013333 ± 0.015422
- recovery_success_rate: 0.860000 ± 0.010458
- peak_separation: 0.669160 ± 0.238171

### disturbed_time
- alert_precision: 0.824984 ± 0.036213
- alert_recall: 0.942500 ± 0.052738
- baseline_stability_rate: 0.991250 ± 0.001398
- false_positive_rate: 0.015833 ± 0.012977
- recovery_success_rate: 0.867500 ± 0.011180
- peak_separation: 0.853300 ± 0.218592

## Node Role Checks (perturbation alert rate means)

- coherent_time: A=0.0000, B=0.0000, C=0.9450, D=0.0400
- disturbed_time: A=0.0000, B=0.0000, C=0.9425, D=0.0475

## Phase-Aware Confusion Artifact

- CSV: `upgraded_multinode_phase_confusion.csv`
- Contains TP/FP/TN/FN and derived rates for each (condition, node, phase).

