@echo off
title Flappy Evolution Launcher

echo.
echo  ==========================================
echo   Flappy Evolution -- Neural Network Sim
echo  ==========================================
echo.

:: --- Backend ---
echo  [1/2] Starting Python backend (port 8000)...
start "Flappy Backend" cmd /k "cd /d "%~dp0backend" && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

:: Give the backend 3 seconds to bind the port before opening the browser
timeout /t 3 /nobreak > nul

:: --- Frontend ---
echo  [2/2] Starting frontend dev server (port 5173)...
start "Flappy Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

:: Give Vite a few seconds to compile, then open the browser
timeout /t 4 /nobreak > nul

echo  Opening browser...
start "" "http://localhost:5173"

echo.
echo  Both servers are running in separate windows.
echo  Close those windows to stop the servers.
echo.
pause
