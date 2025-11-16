@echo off
cd /d "%~dp0"
chcp 65001 >nul 2>&1

echo ================================================
echo  🔧 Debug - Início dos Serviços
echo ================================================

REM Verificações básicas
if not exist .venv\Scripts\python.exe (
	echo ❌ Ambiente virtual não encontrado!
	pause
	exit /b 1
)

if not exist backend\app\api\main.py (
	echo ❌ main.py não encontrado!
	pause
	exit /b 1
)

echo ✅ Verificações passaram

REM Ativar ambiente e testar dependências
echo 🐍 Ativando ambiente virtual...
call .venv\Scripts\activate.bat
python -c "import fastapi, uvicorn; print('✅ FastAPI OK')" || (echo ❌ FastAPI não instalado & pause & exit /b 1)

echo 📡 Iniciando backend (foreground para debug)...
echo Backend rodará nesta janela. Pressione Ctrl+C para parar.
echo URL: http://localhost:8000
echo.

REM Rodar backend em foreground para ver erros
python -m uvicorn backend.app.api.main:app --host 0.0.0.0 --port 8000 --reload --log-level info

pause