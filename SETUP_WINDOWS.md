# Windows Setup Guide for DIC03-ContAInnum

## Summary of Setup Completed

✅ Python virtual environment configured  
✅ All backend dependencies installed  
✅ Import error in synthesis_manufacturer.py fixed  
✅ Backend server ready to start  
✅ Frontend startup scripts created  

## Quick Start (Everything Installed)

Open **3 separate Command Prompt or PowerShell windows** and run:

### Window 1: Start Ollama (if not already running)
```cmd
ollama serve
```
✅ Already installed and running

### Window 2: Start Backend
Double-click: **`run_backend.cmd`**

Or manually:
```cmd
cd DIC03-ContAInnum
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Backend will be at: http://localhost:8000

### Window 3: Start Frontend
Double-click: **`run_frontend.cmd`**

Or manually:
```cmd
cd DIC03-ContAInnum\frontend
npm install  # First time only
npm run dev
```

Frontend will be at: http://localhost:5173

---

## Prerequisites Installed

### 1. Ollama ✅
- **Status**: Installed (v0.12.11)
- **Mistral Model**: Already pulled
- **Running**: Yes (port 11434)

### 2. Node.js ✅  
- **Status**: Installed (v24.11.1)
- **npm**: Installed (v11.6.2)
- **PATH**: Added to system environment

### 3. Python 3.12 ✅
- **Virtual Environment**: `.venv/`
- **Backend Dependencies**: All installed
- **Flask/FastAPI**: Ready

---

## Fixed Issues

### 1. Import Error in synthesis_manufacturer.py
- **Problem**: Was importing `FreeRetrosynthesisEngine` (didn't exist)
- **Solution**: Changed to `RetrosynthesisEngine` (correct class name)
- **Status**: ✅ Fixed

### 2. Node.js PATH Issue  
- **Problem**: Node.js was installed but not in PATH
- **Solution**: Added `C:\Program Files\nodejs` to system PATH
- **Status**: ✅ Fixed

---

## Testing the Application

### 1. Test Backend Health
```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8000/api/health
```

### 2. Test Ollama Connection
Backend automatically connects to Ollama on startup

### 3. Visit Frontend
Open browser to: http://localhost:5173

---

## Troubleshooting

### Backend won't start
1. Check if port 8000 is in use: `netstat -ano | findstr :8000`
2. Stop any process using port 8000
3. Make sure Ollama is running

### Frontend won't start
1. Make sure Node.js is installed: `node --version`
2. Check if port 5173 is in use
3. Delete `node_modules` and `npm install` again

### npm command not found in PowerShell
- This is a PATH issue. Either:
  - Use cmd.exe instead of PowerShell
  - Or restart PowerShell after the PATH changes
  - Or manually add Node.js to PATH in current session:
    ```powershell
    $env:Path += ";C:\Program Files\nodejs"
    ```

### Ollama not responding
1. Verify Ollama is running: `ollama serve` in a terminal
2. Check if Mistral model is installed: `ollama list`
3. If missing, run: `ollama pull mistral`

---

## File Locations

- Backend code: `DIC03-ContAInnum/backend/`
- Frontend code: `DIC03-ContAInnum/frontend/src/`
- Start backend: `DIC03-ContAInnum/run_backend.cmd`
- Start frontend: `DIC03-ContAInnum/run_frontend.cmd`

---

## Next Steps

1. Double-click `run_backend.cmd` - Start backend server
2. Double-click `run_frontend.cmd` - Start frontend server  
3. Open http://localhost:5173 in browser
4. Use the application!

Enjoy!

