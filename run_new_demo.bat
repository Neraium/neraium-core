@echo off
setlocal
cd /d %~dp0

REM Canonical launcher for the final Tesla UI demo (port 3004).
python run_demo.py --frontend-port 3004 --open

