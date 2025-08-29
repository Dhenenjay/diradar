# Deepfake Intelligence Radar - Bootstrap Script (PowerShell)
# This script sets up the environment and launches the Streamlit app

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "🛰️  Deepfake Intelligence Radar Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
$pythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $pythonCmd = "python3"
} else {
    Write-Host "❌ Error: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.10 or higher and try again." -ForegroundColor Red
    exit 1
}

# Check Python version
Write-Host "🐍 Checking Python version..." -ForegroundColor Yellow
$pythonVersion = & $pythonCmd --version 2>&1
Write-Host "✅ Found $pythonVersion" -ForegroundColor Green

# Check if we're in a virtual environment
if (-not $env:VIRTUAL_ENV) {
    Write-Host ""
    Write-Host "⚠️  Warning: Not running in a virtual environment" -ForegroundColor Yellow
    Write-Host "It's recommended to use a virtual environment." -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Do you want to continue anyway? (y/n)"
    if ($response -ne 'y' -and $response -ne 'Y') {
        Write-Host "Exiting. Please activate a virtual environment and run again." -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "✅ Virtual environment detected: $env:VIRTUAL_ENV" -ForegroundColor Green
}

# Install/upgrade pip
Write-Host ""
Write-Host "📦 Upgrading pip..." -ForegroundColor Yellow
& $pythonCmd -m pip install --upgrade pip --quiet

# Install requirements
Write-Host ""
Write-Host "📦 Installing requirements..." -ForegroundColor Yellow
Write-Host "This may take a few minutes on first run..." -ForegroundColor Gray

if (Test-Path "requirements.txt") {
    & $pythonCmd -m pip install -r requirements.txt
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "❌ Error: Failed to install requirements" -ForegroundColor Red
        Write-Host "Please check the error messages above and try again." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "❌ Error: requirements.txt not found" -ForegroundColor Red
    Write-Host "Please ensure you're running this script from the project directory." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ All requirements installed successfully!" -ForegroundColor Green

# Check if app.py exists
if (-not (Test-Path "app.py")) {
    Write-Host "❌ Error: app.py not found" -ForegroundColor Red
    Write-Host "Please ensure you're running this script from the project directory." -ForegroundColor Red
    exit 1
}

# Launch Streamlit app
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "🚀 Launching Deepfake Intelligence Radar" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📡 Server Configuration:" -ForegroundColor Yellow
Write-Host "   - Port: 7860" -ForegroundColor White
Write-Host "   - Mode: Headless (no browser auto-open)" -ForegroundColor White
Write-Host ""
Write-Host "🌐 Access the application at:" -ForegroundColor Yellow
Write-Host "   - Local:    http://localhost:7860" -ForegroundColor Green
Write-Host "   - Network:  http://$($env:COMPUTERNAME):7860" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Logs:" -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray

# Run Streamlit with specified configuration
& $pythonCmd -m streamlit run app.py `
    --server.port 7860 `
    --server.headless true `
    --server.address 0.0.0.0 `
    --browser.gatherUsageStats false `
    --theme.base "dark" `
    --theme.primaryColor "#FF6B6B" `
    --theme.backgroundColor "#0E1117" `
    --theme.secondaryBackgroundColor "#262730" `
    --theme.textColor "#FAFAFA"

# This will only run if Streamlit exits
Write-Host ""
Write-Host "🛑 Application stopped." -ForegroundColor Red
