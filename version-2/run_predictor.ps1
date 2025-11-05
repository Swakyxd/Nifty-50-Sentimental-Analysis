# NIFTY 50 Market Predictor - PowerShell Launcher
# This script runs the prediction tool with the correct Python environment

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

& F:\anaconda\envs\DL-Project\python.exe predict_market.py $args
