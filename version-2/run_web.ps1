# Launch NIFTY 50 Market Predictor Web Interface
# ==============================================

Write-Host ""
Write-Host "========================================"
Write-Host " NIFTY 50 Market Predictor - Web UI"
Write-Host "========================================"
Write-Host ""
Write-Host "Starting Streamlit web interface..."
Write-Host ""
Write-Host "Access the app at: http://localhost:8501"
Write-Host "Press Ctrl+C to stop the server"
Write-Host ""

# Activate conda environment if available
if (Test-Path "F:\anaconda\shell\condabin\conda-hook.ps1") {
    & "F:\anaconda\shell\condabin\conda-hook.ps1"
    conda activate DL-Project
}

# Run streamlit
streamlit run predict_web.py
