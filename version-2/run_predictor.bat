@echo off
REM NIFTY 50 Market Predictor - Interactive Mode Launcher
REM This batch file runs the prediction script with the correct Python environment

cd /d "%~dp0"
F:\anaconda\envs\DL-Project\python.exe predict_market.py %*

pause
