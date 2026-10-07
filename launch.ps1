$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Project environment is missing. Follow README.md installation steps.'
}
$env:PYTHONUTF8 = '1'
& $python -m streamlit run app.py --server.address 127.0.0.1
