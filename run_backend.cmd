@echo off
pushd "%~dp0"
REM Add Node.js to PATH for this session
set PATH=C:\Program Files\nodejs;%PATH%
REM Start uvicorn without reload to avoid watching node_modules
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
popd

pause
