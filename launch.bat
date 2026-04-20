@echo off
REM NeRAIUM Demo Launch Script (Windows)
REM Starts backend and frontend with one command

echo.
echo 🚀 Starting NeRAIUM System Intelligence Interface...
echo.

REM Check if we're in the right directory
if not exist "frontend" (
    echo ❌ Error: Script must be run from project root directory
    exit /b 1
)

if not exist "apps\api" (
    echo ❌ Error: Script must be run from project root directory
    exit /b 1
)

REM Start backend in new window
echo Starting Backend Server...
start "NeRAIUM Backend" cmd /k python -m uvicorn apps.api.main:app --reload --port 8000
echo ✓ Backend started at http://localhost:8000
echo.

REM Wait for backend to initialize
timeout /t 3 /nobreak

REM Start frontend in new window
echo Starting Frontend Server...
cd frontend
start "NeRAIUM Frontend" cmd /k npm run dev
echo ✓ Frontend started at http://localhost:3000
echo.

echo.
echo ========================================
echo NeRAIUM System Intelligence is running!
echo ========================================
echo.
echo 📊 Dashboard:     http://localhost:3000
echo 🔧 API:           http://localhost:8000
echo.
echo Close these windows to stop the servers
echo.

pause
