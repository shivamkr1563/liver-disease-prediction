# Start Backend Server
Write-Host "Starting FastAPI Backend..." -ForegroundColor Green
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd backend; python app.py"

# Wait for backend to start
Start-Sleep -Seconds 3

# Start Frontend Development Server
Write-Host "Starting React Frontend..." -ForegroundColor Green
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev"

Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "🚀 Application Starting!" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Backend API: http://localhost:8000" -ForegroundColor Yellow
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Yellow
Write-Host "==================================================" -ForegroundColor Cyan
