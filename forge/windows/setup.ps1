# MLO Forge: one-time setup check on Windows.
#   powershell -ExecutionPolicy Bypass -File forge\windows\setup.ps1
# Checks Python, installs the forge's Python packages, finds Blender and checks Sollumz/PyMateria.
# It does NOT install Blender, Sollumz, CodeWalker or FiveM for you; it tells you what is missing.

$ErrorActionPreference = "Stop"
$forge = Split-Path -Parent $PSScriptRoot

Write-Host "== Python" -ForegroundColor Cyan
$py = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $py) { Write-Host "Python 3.10+ missing: install from python.org (tick 'Add to PATH')" -ForegroundColor Red; exit 1 }
python --version
python -m pip install --upgrade -r "$forge\requirements.txt"

Write-Host "== Blender" -ForegroundColor Cyan
$candidates = @($env:BLENDER) + (Get-ChildItem "C:\Program Files\Blender Foundation\*\blender.exe" -ErrorAction SilentlyContinue |
    Sort-Object FullName -Descending | ForEach-Object { $_.FullName })
$blender = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $blender) { Write-Host "Blender 5.x not found: install from blender.org, or set BLENDER=<path to blender.exe>" -ForegroundColor Red; exit 1 }
Write-Host "using $blender"
[Environment]::SetEnvironmentVariable("BLENDER", $blender, "User")

Write-Host "== Sollumz / PyMateria" -ForegroundColor Cyan
& $blender -b --python "$forge\blender\check_sollumz.py" 2>&1 | Select-String "CHECK|→"

Write-Host ""
Write-Host "Next:" -ForegroundColor Green
Write-Host "  cd $forge"
Write-Host "  python -m forge.build specs\building_001.json"
Write-Host "  powershell -File windows\install_resource.ps1 -Server C:\FXServer\server-data"
