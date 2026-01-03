# Test simple para subir PDF y monitorear status
$API_URL = "http://localhost:8000/api/pdf"
$UPLOAD_FILE = "uploads\1478_47-10-01-017_C__372_pag___7a682a374769c22c.pdf"

Write-Host "===== TEST SIMPLE DE CARGA Y MONITOREO =====" -ForegroundColor Green
Write-Host ""

# Paso 1: Subir PDF
Write-Host "PASO 1: Subiendo PDF..."
$uploadResponse = curl.exe -s -X POST `
  -F "file=@$UPLOAD_FILE" `
  -F "use_ocr=true" `
  "$API_URL/upload" | ConvertFrom-Json

$id = $uploadResponse.id
$task_id = $uploadResponse.task_id
$status = $uploadResponse.status

Write-Host "ID: $id" -ForegroundColor Cyan
Write-Host "Task: $task_id" -ForegroundColor Cyan
Write-Host "Status: $status" -ForegroundColor Yellow
Write-Host ""

# Paso 2: Monitorear status
Write-Host "PASO 2: Monitoreando estado (esperando procesamiento)..."
for ($i = 1; $i -le 10; $i++) {
    Start-Sleep -Seconds 2
    
    $statusResponse = curl.exe -s "$API_URL/upload-status?pdf_id=$id"
    
    if ($statusResponse) {
        $statusData = $statusResponse | ConvertFrom-Json
        $current_status = $statusData.status
        $progress = $statusData.progress
        $pages = $statusData.pages
        
        Write-Host "[$i] Status: $current_status | Progress: $progress% | Pages: $pages"
        
        if ($current_status -eq "completed") {
            Write-Host "✅ COMPLETADO!" -ForegroundColor Green
            break
        }
    }
}

Write-Host ""
Write-Host "Test finalizado."
