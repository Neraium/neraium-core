#!/usr/bin/env bash
set -euo pipefail

CORPUS_SET=${CORPUS_SET:-"baseline_clean,noisy_realistic,adversarial,transfer_cross_domain"}
HISTORY_ROOT=${HISTORY_ROOT:-"reports/validation/history"}

python tools/run_release_candidate.py \
  --corpus-set "${CORPUS_SET}" \
  --history-root "${HISTORY_ROOT}" \
  --output "reports/validation/history/release_candidate/latest_real_world_validation_report.json" \
  --core-output "reports/validation/history/release_candidate/latest_core_validation_report.json" \
  --release-gate-output "reports/validation/history/release_candidate/latest_release_gate_report.json" \
  --multi-corpus-output "reports/validation/history/release_candidate/latest_multi_corpus_release_report.json"

python -m validation.history.trends --history-root "${HISTORY_ROOT}" --output "${HISTORY_ROOT}/trend_summary.json"
