# NIFTY 50 Market Predictor Web App Launcher
# This script activates the DL-GPU environment and starts the Flask web server

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " NIFTY 50 Market Predictor Web App" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Activate conda environment
Write-Host "Activating DL-GPU environment..." -ForegroundColor Yellow
& F:\anaconda\shell\condabin\conda-hook.ps1
conda activate DL-GPU

# Check if Flask is installed
Write-Host "Checking Flask installation..." -ForegroundColor Yellow
python -c "import flask" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Flask is not installed. Installing now..." -ForegroundColor Yellow
    pip install flask
}

# Start the web app
Write-Host ""
Write-Host "Starting web server..." -ForegroundColor Green
Write-Host ""
python app.py
