@echo off
REM Start Frontend Development Server
cd /d "%~dp0frontend"

REM Add Node.js to PATH
set PATH=C:\Program Files\nodejs;%PATH%

echo.
echo === DIC03-ContAInnum Frontend Server ===
echo.

REM Check if node_modules exists
if not exist "node_modules" (
    echo Installing npm dependencies...
    call npm install
    if errorlevel 1 (
        echo ERROR: npm install failed
        pause
        exit /b 1
    )
)

echo.
echo Starting Vite development server on http://localhost:5173
echo Press Ctrl+C to stop
echo.

call npm run dev
pause
