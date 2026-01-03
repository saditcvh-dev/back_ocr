$API_URL = "http://localhost:8000/api/pdf"

Write-Host "=== TEST 1: Verificar que API responde ===" -ForegroundColor Green
$result = curl.exe -X GET "$API_URL/list" -H "Content-Type: application/json" -s
Write-Host $result | ConvertFrom-Json | ConvertTo-Json -Depth 2

Write-Host "`n=== TEST 2: Upload PDF ===" -ForegroundColor Green
$pdfPath = "uploads\1478_47-10-01-017_C__372_pag___7a682a374769c22c.pdf"
if (Test-Path $pdfPath) {
    Write-Host "PDF encontrado: $pdfPath"
    
    # Upload con curl multipart
    $response = curl.exe -X POST "$API_URL/upload" `
        -F "file=@$pdfPath" `
        -F "use_ocr=true" `
        -s
    
    $jsonResponse = $response | ConvertFrom-Json
    Write-Host "Response:" -ForegroundColor Cyan
    Write-Host ($jsonResponse | ConvertTo-Json -Depth 3)
    
    $taskId = $jsonResponse.task_id
    $pdfId = $jsonResponse.id
    
    if ($taskId) {
        Write-Host "`n=== TEST 3: Monitorear status (5 intentos) ===" -ForegroundColor Green
        for ($i = 1; $i -le 5; $i++) {
            Start-Sleep -Seconds 2
            $statusUrl = "$API_URL/upload-status/$pdfId"
            $status = curl.exe -X GET $statusUrl -H "Content-Type: application/json" -s | ConvertFrom-Json
            
            Write-Host "[$i] Status: $($status.status) | Progress: $($status.progress)% | Pages: $($status.pages)" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "PDF no encontrado en: $pdfPath" -ForegroundColor Red
}

Write-Host "`n=== TERMINADO ===" -ForegroundColor Green
