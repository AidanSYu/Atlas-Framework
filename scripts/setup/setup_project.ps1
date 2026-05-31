# Nuke-and-Pave bootstrap for the Atlas Framework backend + frontend.
# Creates a fresh .venv with CUDA 12.1 GPU builds of torch + llama-cpp-python,
# then installs the CORE backend requirements and the JS toolchains.
#
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File scripts\setup\setup_project.ps1
# Optional plugin stacks (install afterwards, only if you need them):
#   pip install -r src\backend\requirements.txt -r src\backend\requirements.chemistry.txt
#   pip install -r src\backend\requirements.txt -r src\backend\requirements.prometheus.txt

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$VenvPath = Join-Path $ProjectRoot ".venv"

Write-Host "=== Step 1: Remove existing .venv (if any) ===" -ForegroundColor Cyan
if (Test-Path $VenvPath) {
    Remove-Item -Recurse -Force $VenvPath
    Write-Host "Removed .venv" -ForegroundColor Green
}
else {
    Write-Host "No .venv found, skipping." -ForegroundColor Yellow
}

Write-Host "`n=== Step 2: Create fresh virtual environment ===" -ForegroundColor Cyan
Set-Location $ProjectRoot
python -m venv .venv
if (-not $?) { throw "Failed to create venv" }
& (Join-Path $VenvPath "Scripts\Activate.ps1")
python -m pip install --upgrade pip
Write-Host "Created and activated .venv" -ForegroundColor Green

Write-Host "`n=== Step 3: GPU runtime — CUDA 12.1 torch + llama-cpp-python ===" -ForegroundColor Cyan
# Install the CUDA wheels from their dedicated indexes BEFORE requirements.txt so
# the pins in requirements.txt are already satisfied and never fall back to CPU.
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
if (-not $?) { throw "Failed to install CUDA torch" }
pip install llama-cpp-python==0.3.19 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
if (-not $?) {
    Write-Host "Prebuilt CUDA wheel unavailable - building from source (needs CUDA toolkit + MSVC)" -ForegroundColor Yellow
    $env:CMAKE_ARGS = "-DGGML_CUDA=on"
    pip install llama-cpp-python==0.3.19 --no-cache-dir
    if (-not $?) { throw "Failed to install CUDA llama-cpp-python" }
}
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
Write-Host "GPU runtime installed. If 'CUDA available: False' above, your llama/torch fell back to CPU." -ForegroundColor Yellow

Write-Host "`n=== Step 4: Install CORE backend requirements ===" -ForegroundColor Cyan
# Aliyun mirror for download speed, scoped to THIS command only (no global pip
# config mutation). torch/llama are already satisfied above and are skipped here.
Set-Location (Join-Path $ProjectRoot "src\backend")
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
if (-not $?) { throw "Failed to install requirements.txt" }
Write-Host "Core backend dependencies installed." -ForegroundColor Green

Write-Host "`n=== Step 5: Frontend dependencies ===" -ForegroundColor Cyan
Set-Location (Join-Path $ProjectRoot "src\frontend")
npm install
if (-not $?) { throw "Failed to run npm install in frontend" }
if (-not (Test-Path ".env.local")) {
    Copy-Item ".env.example" ".env.local"
    Write-Host "Created src\frontend\.env.local from .env.example" -ForegroundColor Green
}
Write-Host "Frontend dependencies installed." -ForegroundColor Green

Write-Host "`n=== Step 6: Root / Tauri CLI ===" -ForegroundColor Cyan
Set-Location $ProjectRoot
npm install
if (-not $?) { throw "Failed to run npm install at root" }
Write-Host "Root dependencies (Tauri CLI, etc.) installed." -ForegroundColor Green

Write-Host "`n=== Done. ===" -ForegroundColor Green
Write-Host "Next:" -ForegroundColor Gray
Write-Host "  1. Copy config\.env.example -> config\.env and add API keys (optional)" -ForegroundColor Gray
Write-Host "  2. Place GGUF models in models\ (orchestrator + nomic-embed + gliner) - they are gitignored" -ForegroundColor Gray
Write-Host "  3. Activate venv:  .\.venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host "  4. cd src\backend; python run_server.py" -ForegroundColor Gray
