#!/bin/bash

# NeRAIUM Demo Launch Script (Git Bash Compatible)
# Starts backend and frontend

echo ""
echo "🚀 Starting NeRAIUM System Intelligence Interface..."
echo ""

# Get project root
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# Start backend in background
echo "Starting Backend Server..."
python -m uvicorn apps.api.main:app --reload --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!
echo "✓ Backend started (PID: $BACKEND_PID)"
sleep 3

# Start frontend in background
echo "Starting Frontend Server..."
cd frontend
npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "✓ Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "========================================="
echo "NeRAIUM is running!"
echo "========================================="
echo ""
echo "📊 Dashboard:     http://localhost:3000"
echo "🔧 API:           http://localhost:8000"
echo ""
echo "Logs:"
echo "  Backend:  tail -f backend.log"
echo "  Frontend: tail -f frontend/frontend.log"
echo ""
echo "Stop: kill $BACKEND_PID $FRONTEND_PID"
echo ""

# Keep script running
wait
