# Fully automated restoration script for Atlas 2.0
$ErrorActionPreference = "Continue"

Write-Host "=== Sub-Task 2: Installing Remaining Global Dependencies ===" -ForegroundColor Cyan

Write-Host "Installing Rust (via up-init/winget)..."
winget install --id Rustlang.Rustup -e --source winget --accept-source-agreements --accept-package-agreements

Write-Host "Installing MSVC C++ Build Tools (requires Admin/UAC)..."
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --accept-source-agreements --accept-package-agreements --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

# Refresh environment variables locally for this script so we can use Node and Rust right after
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host "`n=== Sub-Task 3: Running Project Setup Script ===" -ForegroundColor Cyan
$ProjectSetupScript = Join-Path $PSScriptRoot "setup_project.ps1"
if (Test-Path $ProjectSetupScript) {
    & $ProjectSetupScript
} else {
    Write-Host "Could not find setup_project.ps1" -ForegroundColor Red
}

Write-Host "`n=== Final Git Status Check ===" -ForegroundColor Cyan
git status
if ($?) {
    Write-Host "All good! The repository is ready." -ForegroundColor Green
} else {
    Write-Host "Git status failed (expected if git hasn't propagated to PATH). Restart your terminal!" -ForegroundColor Yellow
}
