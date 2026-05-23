@echo off
setlocal

set "BASE_DIR=%~dp0"
set "PYTHON_EXE=D:\anaconda\envs\rasapro\python.exe"
set "SANIC_RESPONSE_TIMEOUT=500"

echo ================================
echo Starting ECS Services (rasapro)
echo ================================

REM === Rasa Core ===
start "Rasa Core" /D "%BASE_DIR%rasa" cmd /k call "%PYTHON_EXE%" -m rasa run --enable-api

REM === Rasa Actions ===
REM Image query may be slow, SANIC_RESPONSE_TIMEOUT is set above.
start "Rasa Actions" /D "%BASE_DIR%rasa" cmd /k call "%PYTHON_EXE%" -m rasa run actions

REM === FastAPI ===
start "FastAPI" /D "%BASE_DIR%" cmd /k call "%PYTHON_EXE%" -m uvicorn web.main:app --reload --port 8000

REM === Embedding Service ===
start "Embedding Service" /D "%BASE_DIR%rasa\addons" cmd /k call "%PYTHON_EXE%" -m uvicorn embed_service:app --port 10010

echo All services started.
