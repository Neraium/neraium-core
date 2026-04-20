#!/bin/bash

# NeRAIUM Demo Launch Script
# Starts backend and frontend with one command

set -e

echo "🚀 Starting NeRAIUM System Intelligence Interface..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if we're in the right directory
if [ ! -d "frontend" ] || [ ! -d "apps/api" ]; then
    echo "❌ Error: Script must be run from project root directory"
    exit 1
fi

# Start backend
echo -e "${BLUE}Starting Backend Server...${NC}"
python -m uvicorn apps.api.main:app --reload --port 8000 &
BACKEND_PID=$!
echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"
echo "  Available at: http://localhost:8000"
echo ""

# Wait for backend to start
sleep 3

# Start frontend
echo -e "${BLUE}Starting Frontend Server...${NC}"
cd frontend
PORT=3002 npm run dev &
FRONTEND_PID=$!
echo -e "${GREEN}✓ Frontend started (PID: $FRONTEND_PID)${NC}"
echo "  Available at: http://localhost:3002"
echo ""

# Print summary
echo -e "${GREEN}========================================${NC}"
echo "NeRAIUM System Intelligence is running!"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "📊 Dashboard:     http://localhost:3000"
echo "🔧 API:           http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Function to kill both processes on exit
cleanup() {
    echo ""
    echo -e "${BLUE}Shutting down...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo -e "${GREEN}✓ Servers stopped${NC}"
}

trap cleanup EXIT

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
