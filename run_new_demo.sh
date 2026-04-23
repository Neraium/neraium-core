#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Canonical launcher for the final Tesla UI demo (port 3004).
python run_demo.py --frontend-port 3004 --open

