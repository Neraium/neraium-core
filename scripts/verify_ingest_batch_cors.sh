#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8010}"
ORIGIN="${2:-https://api.example.com}"

echo "== Preflight OPTIONS with browser tracing headers =="
curl -i -X OPTIONS \
  "${BASE_URL}/ingest/batch?customer_id=default-customer&run_id=run_4" \
  -H "Origin: ${ORIGIN}" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: content-type,baggage,sentry-trace,traceparent,tracestate"

echo
echo "== POST /ingest/batch (app-level response; should include Access-Control-Allow-Origin) =="
curl -i -X POST \
  "${BASE_URL}/ingest/batch?customer_id=default-customer&run_id=run_4" \
  -H "Origin: ${ORIGIN}" \
  -H "Content-Type: application/json" \
  --data '{"items":[{"timestamp":"2026-03-24T00:00:00Z","site_id":"demo-site","asset_id":"demo-asset","sensor_values":{"pressure":42,"flow":27,"vibration":6.2,"temperature":61.5}}]}'
