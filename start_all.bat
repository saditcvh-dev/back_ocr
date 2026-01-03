
@echo off
REM Script para iniciar FastAPI, Celery Worker y Redis simultáneamente
REM Este script abre tres ventanas PowerShell independientes

echo.
echo ====================================================
echo INICIANDO SISTEMA PDF API CON CELERY + REDIS
echo ====================================================
echo.

REM Obtener la ruta del proyecto
set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

REM Iniciar Redis
echo [1/3] Iniciando Redis (Docker)...
docker start redis-pdf-api 2>nul
if errorlevel 1 (
    echo ⚠️  Redis no encontrado. Asegúrate que Docker esté corriendo.
    echo Ejecuta manualmente: docker start redis-pdf-api
)

REM Esperar 2 segundos
timeout /t 2 /nobreak

REM Iniciar FastAPI en una nueva ventana
echo [2/3] Iniciando FastAPI en puerto 8000...
start "FastAPI Server" powershell -NoExit -Command "cd '%PROJECT_DIR%'; .\venv\Scripts\Activate.ps1; python run.py"

REM Esperar 3 segundos para que FastAPI se inicie
timeout /t 3 /nobreak

REM Iniciar Celery Worker en otra ventana
echo [3/3] Iniciando Celery Worker...
start "Celery Worker" powershell -NoExit -Command "cd '%PROJECT_DIR%'; .\venv\Scripts\Activate.ps1; celery -A app.core.celery_app worker --loglevel=info --concurrency=1"

echo.
echo ====================================================
echo ✅ SERVICIOS INICIADOS
echo ====================================================
echo.
echo 🚀 FastAPI:   http://localhost:8000
echo 🐰 Celery:    Conectado a Redis
echo 📦 Redis:     localhost:6379 (Docker)
echo.
echo 📝 Frontend:  http://localhost:4200
echo.
echo ✋ Para detener: Cierra las tres ventanas de PowerShell
echo.
pause

