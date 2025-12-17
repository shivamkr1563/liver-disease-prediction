# Test Backend API Health
Write-Host "Testing Backend API..." -ForegroundColor Green

try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
    Write-Host "✓ Backend is healthy!" -ForegroundColor Green
    $response | ConvertTo-Json
} catch {
    Write-Host "✗ Backend is not responding" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`nTesting Prediction Endpoint..." -ForegroundColor Green

$testData = @{
    age = 45
    gender = 1
    total_bilirubin = 0.9
    direct_bilirubin = 0.3
    alkaline_phosphotase = 202
    alamine_aminotransferase = 32
    aspartate_aminotransferase = 35
    total_proteins = 7.2
    albumin = 4.1
    albumin_and_globulin_ratio = 1.32
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $testData -ContentType "application/json"
    Write-Host "✓ Prediction successful!" -ForegroundColor Green
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "✗ Prediction failed" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}
