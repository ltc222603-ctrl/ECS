@echo off
set BASE_DIR=%~dp0

REM ===== rasapro 的 python 绝对路径 =====
set PYTHON_EXE=D:\anaconda\envs\rasapro\python.exe

echo ================================
echo Starting ECS Services (rasapro)
echo ================================

REM === Rasa Core ===
start "Rasa Core" cmd /k ^
cd /d "%BASE_DIR%rasa" ^&^& ^
"%PYTHON_EXE%" -m rasa run --enable-api

REM === Rasa Actions ===
start "Rasa Actions" cmd /k ^
cd /d "%BASE_DIR%rasa" ^&^& ^
"%PYTHON_EXE%" -m rasa run actions

REM === FastAPI ===
start "FastAPI" cmd /k ^
cd /d "%BASE_DIR%" ^&^& ^
"%PYTHON_EXE%" -m uvicorn web.main:app --reload --port 8000

REM === Embedding Service ===
start "Embedding Service" cmd /k ^
cd /d "%BASE_DIR%rasa\addons" ^&^& ^
"%PYTHON_EXE%" -m uvicorn embed_service:app --port 10010

echo All services started.
