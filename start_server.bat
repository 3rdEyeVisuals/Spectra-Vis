@echo off
REM ============================================================================
REM Spectra Vis - Full Stack Launcher (Windows)
REM Copyright (c) 2025 3rdEyeVisuals
REM ============================================================================

echo ============================================================
echo   Spectra Vis - Tensor Visualization
echo   Copyright (c) 2025 3rdEyeVisuals
echo ============================================================
echo.

REM Get the directory where this script is located
cd /d "%~dp0"

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

REM Check Python version
python --version

REM Check if Node.js is available
where node >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH.
    echo Please install Node.js 18+ and try again.
    pause
    exit /b 1
)

REM Show Node version
echo Node.js version:
node --version

REM Check if backend dependencies are installed
echo.
echo Checking backend dependencies...
python -c "import fastapi, uvicorn" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Installing backend dependencies...
    pip install -r backend\requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to install backend dependencies.
        pause
        exit /b 1
    )
)

REM Check if frontend dependencies are installed
echo Checking frontend dependencies...
if not exist "frontend\node_modules" (
    echo Installing frontend dependencies...
    cd frontend
    npm install
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to install frontend dependencies.
        pause
        exit /b 1
    )
    cd ..
)

echo.
echo ============================================================
echo Starting Spectra Vis...
echo.
echo   Backend API:  http://localhost:8000
echo   Frontend UI:  http://localhost:3000
echo   API Docs:     http://localhost:8000/docs
echo.
echo Press Ctrl+C in each window to stop.
echo ============================================================
echo.

REM Start backend in a new window
start "Spectra Vis - Backend" cmd /k "cd /d %~dp0 && python backend\server.py"

REM Wait a moment for backend to start
timeout /t 2 /nobreak >nul

REM Start frontend in a new window
start "Spectra Vis - Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

REM Open browser after a short delay
timeout /t 4 /nobreak >nul
start http://localhost:3000

echo Both servers started. Browser opening...
echo.
pause
