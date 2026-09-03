@echo off
cd /d "%~dp0backend"
if not exist venv\Scripts\python.exe (
  py -m venv venv
  venv\Scripts\python.exe -m pip install -r requirements.txt
)
venv\Scripts\python.exe -m uvicorn main:app --reload
pause
