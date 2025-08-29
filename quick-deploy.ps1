# Quick Daytona Deployment for Hackathon
Write-Host "🚀 QUICK DEPLOY FOR HACKATHON" -ForegroundColor Green
Write-Host "=============================" -ForegroundColor Green

# Check if running as admin for Docker
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "⚠️  Run PowerShell as Administrator for best results" -ForegroundColor Yellow
}

# Quick local test first
Write-Host "`n📦 Testing locally first..." -ForegroundColor Cyan
Write-Host "Starting Streamlit app on port 7860..." -ForegroundColor White

# Start the app directly
Start-Process -NoNewWindow python -ArgumentList "-m streamlit run app.py --server.port 7860 --server.headless true --server.address 0.0.0.0"

Write-Host "`n✅ App is running!" -ForegroundColor Green
Write-Host "🌐 Access at: http://localhost:7860" -ForegroundColor Cyan
Write-Host "`nPress Ctrl+C to stop" -ForegroundColor Yellow

# Keep script running
while ($true) { Start-Sleep -Seconds 1 }
