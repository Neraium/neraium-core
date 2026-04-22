# FD004 refinement pass evaluation (60 units, max 220 cycles)

## Before vs After key metrics

| Metric | Before | After |
|---|---:|---:|
| advanced_analytics_population_rate | 0.9871120180431747 | 0.9871120180431747 |
| first_alert_cycle_std | 0.25544245225545675 | 0.25544245225545675 |
| first_alert_cycle_unique_count | 2 | 2 |
| first_alert_mode_share | 0.9298245614035088 | 0.9298245614035088 |
| lock_in_saturation_rate | 0.0 | 0.0 |
| imminent_horizon_rate | 0.0 | 0.0 |
| branching_true_rate | n/a | 0.10460745354956504 |
| lock_in p90 | 0.5103 | 0.2739 |
| transition_pressure p90 | 0.583 | 0.5263 |

### Horizon distribution
- before: {"LONGER_HORIZON": 0.833637632907314, "MID_TERM": 0.10643325099344861, "NEAR_TERM": 0.0470411341424122, "UNKNOWN": 0.012887981956825261}
- after: {"LONGER_HORIZON": 0.9074213296101385, "MID_TERM": 0.03393835248630652, "NEAR_TERM": 0.04575233594672967, "UNKNOWN": 0.012887981956825261}

### Branch count estimate quantiles (after)
- {"0.1": 1.0, "0.25": 1.0, "0.5": 1.0, "0.75": 2.0, "0.9": 2.0}

## Sample units (after)
### unit 36 (first ALERT cycle: 25)
| cycle | state | transition_pressure | lock_in | horizon | is_branching | branch_count |
|---:|---|---:|---:|---|---|---:|
| 20 | STABLE | 0.0560 | 0.0298 | LONGER_HORIZON | False | 1.0 |
| 24 | WATCH | 0.6705 | 0.1446 | LONGER_HORIZON | False | 1.0 |
| 25 | ALERT | 0.7505 | 0.3238 | MID_TERM | False | 1.0 |
| 26 | ALERT | 0.7764 | 0.4407 | MID_TERM | False | 1.0 |
| 30 | ALERT | 0.7840 | 0.6053 | NEAR_TERM | True | 2.0 |
| 40 | WATCH | 0.6220 | 0.2272 | LONGER_HORIZON | True | 2.0 |
| 60 | STABLE | 0.4734 | 0.2116 | LONGER_HORIZON | False | 2.0 |
| 100 | STABLE | 0.3780 | 0.1164 | LONGER_HORIZON | False | 1.0 |
| 150 | STABLE | 0.3078 | 0.1964 | LONGER_HORIZON | False | 2.0 |

### unit 50 (first ALERT cycle: 25)
| cycle | state | transition_pressure | lock_in | horizon | is_branching | branch_count |
|---:|---|---:|---:|---|---|---:|
| 20 | STABLE | 0.0560 | 0.0298 | LONGER_HORIZON | False | 1.0 |
| 24 | WATCH | 0.6705 | 0.1647 | LONGER_HORIZON | False | 1.0 |
| 25 | ALERT | 0.7505 | 0.3478 | MID_TERM | False | 1.0 |
| 26 | WATCH | 0.6684 | 0.4300 | LONGER_HORIZON | False | 1.0 |
| 30 | ALERT | 0.8542 | 0.5155 | MID_TERM | True | 2.0 |
| 40 | STABLE | 0.3321 | 0.2118 | LONGER_HORIZON | True | 2.0 |
| 60 | STABLE | 0.3780 | 0.1181 | LONGER_HORIZON | False | 1.0 |
| 100 | STABLE | 0.3078 | 0.1942 | LONGER_HORIZON | False | 2.0 |
| 150 | STABLE | 0.3078 | 0.1161 | LONGER_HORIZON | False | 1.0 |
| 200 | STABLE | 0.3456 | 0.1978 | LONGER_HORIZON | False | 1.0 |

### unit 17 (first ALERT cycle: 25)
| cycle | state | transition_pressure | lock_in | horizon | is_branching | branch_count |
|---:|---|---:|---:|---|---|---:|
| 20 | STABLE | 0.0560 | 0.0298 | LONGER_HORIZON | False | 1.0 |
| 24 | WATCH | 0.6705 | 0.1723 | LONGER_HORIZON | False | 1.0 |
| 25 | ALERT | 0.7505 | 0.3478 | MID_TERM | False | 1.0 |
| 26 | WATCH | 0.6684 | 0.4295 | LONGER_HORIZON | False | 1.0 |
| 30 | ALERT | 0.7462 | 0.6020 | NEAR_TERM | True | 2.0 |
| 40 | STABLE | 0.3455 | 0.4272 | MID_TERM | True | 2.0 |
| 60 | STABLE | 0.3078 | 0.1971 | LONGER_HORIZON | False | 1.0 |
| 100 | STABLE | 0.4158 | 0.1384 | LONGER_HORIZON | False | 2.0 |
| 150 | STABLE | 0.3780 | 0.1977 | LONGER_HORIZON | False | 1.0 |
| 200 | STABLE | 0.3780 | 0.1189 | LONGER_HORIZON | False | 1.0 |

### unit 4 (first ALERT cycle: 25)
| cycle | state | transition_pressure | lock_in | horizon | is_branching | branch_count |
|---:|---|---:|---:|---|---|---:|
| 20 | STABLE | 0.0560 | 0.0298 | LONGER_HORIZON | False | 1.0 |
| 24 | WATCH | 0.6705 | 0.1538 | LONGER_HORIZON | False | 1.0 |
| 25 | ALERT | 0.7505 | 0.3478 | MID_TERM | False | 1.0 |
| 26 | WATCH | 0.6684 | 0.4295 | LONGER_HORIZON | False | 1.0 |
| 30 | ALERT | 0.8164 | 0.6082 | NEAR_TERM | True | 2.0 |
| 40 | STABLE | 0.3322 | 0.1771 | LONGER_HORIZON | True | 2.0 |
| 60 | STABLE | 0.5038 | 0.1436 | LONGER_HORIZON | False | 2.0 |
| 100 | STABLE | 0.3456 | 0.1934 | LONGER_HORIZON | False | 1.0 |
| 150 | STABLE | 0.3780 | 0.1172 | LONGER_HORIZON | False | 1.0 |
| 200 | STABLE | 0.4158 | 0.2007 | LONGER_HORIZON | False | 2.0 |

### unit 33 (first ALERT cycle: 25)
| cycle | state | transition_pressure | lock_in | horizon | is_branching | branch_count |
|---:|---|---:|---:|---|---|---:|
| 20 | STABLE | 0.0560 | 0.0298 | LONGER_HORIZON | False | 1.0 |
| 24 | WATCH | 0.6705 | 0.1470 | LONGER_HORIZON | False | 1.0 |
| 25 | ALERT | 0.7505 | 0.3141 | MID_TERM | False | 1.0 |
| 26 | ALERT | 0.7764 | 0.4407 | MID_TERM | False | 1.0 |
| 30 | ALERT | 0.8379 | 0.6131 | NEAR_TERM | True | 2.0 |
| 40 | STABLE | 0.3841 | 0.1718 | LONGER_HORIZON | True | 2.0 |
| 60 | STABLE | 0.3078 | 0.1186 | LONGER_HORIZON | False | 2.0 |
| 100 | STABLE | 0.4158 | 0.1994 | LONGER_HORIZON | False | 2.0 |
