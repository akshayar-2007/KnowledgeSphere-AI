@echo off
title KnowledgeSphere AI Launcher
cd /d "%~dp0"

echo ===================================================
echo             KnowledgeSphere AI Launcher           
echo ===================================================
echo.

:: Check if virtual environment exists, if not create it
if not exist "venv" (
    echo [1/3] Creating virtual environment...
    py -3 -m venv venv
    if errorlevel 1 (
        echo Error: Could not create venv. Make sure Python is installed.
        pause
        exit /b
    )
)

:: Activate venv and install requirements
echo [2/3] Checking/Installing dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install requirements.
    pause
    exit /b
)

:: Start the application
echo.
echo [3/3] Starting KnowledgeSphere AI server...
echo.
echo ===================================================
echo  Access the portal locally at: http://127.0.0.1:5000
echo  Press Ctrl+C in this window to stop the server.
echo ===================================================
echo.
python app.py

pause
