#!/bin/bash

# NeRAIUM Demo Launch Script (Git Bash Compatible)
# Properly backgrounds both backend and frontend

echo ""
echo "🚀 Starting NeRAIUM System Intelligence Interface..."
echo ""

# Get project root
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# Start backend in background (redirect output)
echo "Starting Backend Server..."
python -m uvicorn apps.api.main:app --reload --port 8000 > /tmp/neraium-backend.log 2>&1 &
BACKEND_PID=$!
echo "✓ Backend started on http://localhost:8000 (PID: $BACKEND_PID)"

# Wait for backend to be ready
sleep 3

# Start frontend in background
echo "Starting Frontend Server..."
cd frontend
npm run dev > /tmp/neraium-frontend.log 2>&1 &
FRONTEND_PID=$!
echo "✓ Frontend started on http://localhost:3000 (PID: $FRONTEND_PID)"

echo ""
echo "========================================="
echo "✨ NeRAIUM is running!"
echo "========================================="
echo ""
echo "📊 Open:    http://localhost:3000"
echo "🔧 API:     http://localhost:8000"
echo ""
echo "To stop:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo ""
