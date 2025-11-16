@echo off
REM Script rápido para desenvolvimento - inicia apenas o backend
cd /d "%~dp0"
chcp 65001 >nul 2>&1

echo ⚡ Iniciando apenas o backend para desenvolvimento...

if not exist .venv\Scripts\python.exe (
	echo ❌ Ambiente virtual não encontrado! Execute start.bat primeiro.
	pause
	exit /b 1
)

echo 📡 Backend FastAPI iniciando...
call .venv\Scripts\activate.bat
set PYTHONIOENCODING=utf-8
python -m uvicorn backend.app.api.main:app --host 0.0.0.0 --port 8000 --reload --log-level info