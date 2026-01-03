# Script PowerShell para iniciar FastAPI y Celery
# Uso: .\start_all.ps1

Write-Host ""
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "INICIANDO SISTEMA PDF API CON CELERY + REDIS" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

$ProjectDir = Get-Location
Write-Host "[1/2] Iniciando FastAPI en puerto 8000..." -ForegroundColor Yellow
$FastAPICmd = "cd '$ProjectDir'; & '.\venv\Scripts\Activate.ps1'; python run.py"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $FastAPICmd -WindowStyle Normal

Write-Host "[2/2] Esperando 3 segundos..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

Write-Host "[3/2] Iniciando Celery Worker..." -ForegroundColor Yellow
$CeleryCmd = "cd '$ProjectDir'; & '.\venv\Scripts\Activate.ps1'; celery -A app.core.celery_app worker --loglevel=info --concurrency=1"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $CeleryCmd -WindowStyle Normal

Write-Host ""
Write-Host "====================================================" -ForegroundColor Green
Write-Host "✅ SERVICIOS INICIADOS" -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 FastAPI:   http://localhost:8000" -ForegroundColor Cyan
Write-Host "🐰 Celery:    Conectado a Redis" -ForegroundColor Cyan
Write-Host "📦 Redis:     localhost:6379 (Docker)" -ForegroundColor Cyan
Write-Host ""
Write-Host "📝 Frontend:  http://localhost:4200" -ForegroundColor Cyan
Write-Host ""
Write-Host "✋ Para detener: Cierra ambas ventanas de PowerShell" -ForegroundColor Yellow
Write-Host ""
Read-Host "Presiona ENTER para cerrar esta ventana"
