#!/bin/bash
# Script para iniciar FastAPI y Celery Worker simultáneamente (Linux/Mac)

echo ""
echo "===================================================="
echo "INICIANDO SISTEMA PDF API CON CELERY + REDIS"
echo "===================================================="
echo ""

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Activar venv
source venv/bin/activate

# Iniciar FastAPI en background
echo "[1/2] Iniciando FastAPI en puerto 8000..."
python run.py > logs/fastapi.log 2>&1 &
FASTAPI_PID=$!

# Esperar 3 segundos
sleep 3

# Iniciar Celery Worker en background
echo "[2/2] Iniciando Celery Worker..."
celery -A app.core.celery_app worker --loglevel=info --concurrency=1 > logs/celery.log 2>&1 &
CELERY_PID=$!

echo ""
echo "===================================================="
echo "✅ SERVICIOS INICIADOS"
echo "===================================================="
echo ""
echo "🚀 FastAPI:   http://localhost:8000"
echo "🐰 Celery:    Conectado a Redis (PID: $CELERY_PID)"
echo "📦 Redis:     localhost:6379 (Docker)"
echo ""
echo "📝 Frontend:  http://localhost:4200"
echo ""
echo "Ver logs:"
echo "  - FastAPI:  tail -f logs/fastapi.log"
echo "  - Celery:   tail -f logs/celery.log"
echo ""
echo "Para detener:"
echo "  - kill $FASTAPI_PID $CELERY_PID"
echo ""

# Esperar a que se terminen los procesos
wait
