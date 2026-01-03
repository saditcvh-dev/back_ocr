# Script de prueba: Subir PDF y monitorear estado

$API_BASE = "http://localhost:8000/api/pdf"

Write-Host "`n=== PRUEBA DE API: SUBIR Y MONITOREAR PDF ===" -ForegroundColor Green

# Buscar PDFs en uploads
$pdfs = Get-ChildItem "uploads" -Filter "*.pdf" -ErrorAction SilentlyContinue
if ($pdfs.Count -eq 0) {
    Write-Host "No hay PDFs en uploads/" -ForegroundColor Red
    exit 1
}

$PDF_FILE = $pdfs[0].FullName
Write-Host "Usando PDF: $($pdfs[0].Name)" -ForegroundColor Yellow

# PASO 1: SUBIR EL PDF CON CURL
Write-Host "`nPASO 1: SUBIENDO PDF..." -ForegroundColor Cyan

$response = curl.exe -X POST "$API_BASE/upload?use_ocr=true" -F "file=@`"$PDF_FILE`"" 2>$null

$uploadData = $response | ConvertFrom-Json

$pdfId = $uploadData.id
$taskId = $uploadData.task_id

Write-Host "ID: $pdfId" -ForegroundColor Green
Write-Host "Task: $taskId" -ForegroundColor Green

# PASO 2: MONITOREAR ESTADO
Write-Host "`nPASO 2: MONITOREANDO..." -ForegroundColor Cyan

for ($i = 1; $i -le 20; $i++) {
    
    try {
        $statusResponse = curl.exe -X GET "$API_BASE/upload-status/$pdfId" 2>$null
        
        $statusData = $statusResponse | ConvertFrom-Json
        
        Write-Host "[$i] Status: $($statusData.status) | Progress: $($statusData.progress)% | Pages: $($statusData.pages)" -ForegroundColor White
        
        if ($statusData.status -eq "completed") {
            Write-Host "`nCOMPLETADO!" -ForegroundColor Green
            Write-Host "Paginas: $($statusData.pages)" -ForegroundColor Green
            break
        }
        
    } catch {
        Write-Host "Error: $_" -ForegroundColor Red
    }
    
    Start-Sleep -Seconds 2
}

Write-Host "`nFin de prueba" -ForegroundColor Green
