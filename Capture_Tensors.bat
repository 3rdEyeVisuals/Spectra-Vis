@echo off
title Spectra Vis - Tensor Capture
cd /d "%~dp0"

echo.
echo ============================================================
echo   SPECTRA VIS - Tensor Capture Tool
echo ============================================================
echo.

REM Try to find Python
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python capture_tensors.py
) else (
    echo [ERROR] Python not found in PATH
    echo Please install Python or activate your virtual environment.
    pause
)
