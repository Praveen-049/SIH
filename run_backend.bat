@echo off
cd /d "%~dp0backend"
py -m uvicorn main:app --reload --port 8001
pause
