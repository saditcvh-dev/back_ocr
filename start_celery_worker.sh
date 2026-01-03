#!/bin/bash
# Script para iniciar el worker de Celery en Linux/Mac

# Configurar TESSDATA_PREFIX
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Activar el entorno virtual si existe
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

# Iniciar Celery worker
celery -A app.core.celery_app worker --loglevel=info --concurrency=2
