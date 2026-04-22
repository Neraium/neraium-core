#!/usr/bin/env bash
set -euo pipefail

export NERAIUM_REPO_ROOT="$(pwd)"
export PYTHONPATH="$(pwd)"

echo "Neraium Demo — FastAPI + Next.js"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3004"

python run_demo.py &
BACKEND_PID=$!

cleanup() {
kill $BACKEND_PID 2>/dev/null || true
}
trap cleanup EXIT

cd frontend
npm install
npm run dev -- -p 3004
