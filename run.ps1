# run.ps1 — Helper script for Streaming Live RAG
# Uses the venv at C:\Users\lucky\.ragvenv (outside OneDrive to avoid WDAC DLL blocking)

param(
    [Parameter(Position=0)]
    [string]$Command = "server"
)

$PYTHON = "C:\Users\lucky\.ragvenv\Scripts\python.exe"
$ROOT = $PSScriptRoot

switch ($Command) {
    "server" {
        Write-Host "[StreamRAG] Starting server..." -ForegroundColor Cyan
        & $PYTHON -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
    }
    "data" {
        Write-Host "[StreamRAG] Generating mock data..." -ForegroundColor Cyan
        & $PYTHON scripts/generate_mock_data.py
    }
    "test" {
        Write-Host "[StreamRAG] Running tests..." -ForegroundColor Cyan
        & $PYTHON -m pytest tests/ -v
    }
    "evaluate" {
        Write-Host "[StreamRAG] Running gating checks..." -ForegroundColor Cyan
        & $PYTHON scripts/evaluate.py
    }
    "install" {
        Write-Host "[StreamRAG] Installing dependencies to local venv..." -ForegroundColor Cyan
        & $PYTHON -m pip install -r requirements.txt
    }
    default {
        Write-Host "Usage: .\run.ps1 [server|data|test|evaluate|install]" -ForegroundColor Yellow
    }
}
