$ErrorActionPreference = "Stop"

Write-Host "=== Venus / Magellan V2 ===" -ForegroundColor Cyan
python main_v2.py features
python main_v2.py discover
python main_v2.py interpret

Write-Host "V2 pipeline completed." -ForegroundColor Green
