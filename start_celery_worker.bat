@echo off
REM Script para iniciar el worker de Celery

REM Configurar TESSDATA_PREFIX
set TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata

REM Activar el entorno virtual si existe
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat

REM Iniciar Celery worker
celery -A app.core.celery_app worker --loglevel=info --concurrency=2

pause
