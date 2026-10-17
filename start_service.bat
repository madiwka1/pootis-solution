@echo off
cd /d %~dp0
if "%YOLO_WEIGHTS%"=="" set YOLO_WEIGHTS=%~dp0weights\best.pt
set PYTHONUTF8=1
set PY=python
if exist "%~dp0.venv\Scripts\python.exe" set PY=%~dp0.venv\Scripts\python.exe
"%PY%" -m uvicorn service.app:app --host 0.0.0.0 --port 8000
pause
