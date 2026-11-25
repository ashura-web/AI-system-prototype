@echo off
REM =====================================================
REM  Heart Disease Risk Prediction System
REM  Author: Ar Kar Phyo
REM  Double-click this file: it installs the packages
REM  (first run only), rebuilds the model file for THIS
REM  machine's scikit-learn (2 seconds), and starts the app.
REM =====================================================
title Heart Disease Risk Prediction System
cd /d "%~dp0"

echo [1/3] Checking Python packages...
python -m pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo     Installing requirements - first run only, a few minutes...
    python -m pip install -r requirements.txt
) else (
    echo     Already installed.
)

echo [2/3] Rebuilding heart_model.pkl for THIS machine's scikit-learn...
python make_model.py
if errorlevel 1 (
    echo.
    echo     Model build FAILED - read the message above.
    echo     Most likely heart.csv is missing from this folder.
    pause
    exit /b 1
)

echo [3/3] Starting the web app - your browser will open...
python -m streamlit run app.py

echo.
echo The app was closed. Press any key to exit this window.
pause >nul
