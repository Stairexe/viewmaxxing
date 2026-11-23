# Copy a built resource into a local FiveM server and make sure it starts.
#   powershell -File forge\windows\install_resource.ps1 -Server C:\FXServer\server-data [-Building building_001]
param(
    [Parameter(Mandatory = $true)][string]$Server,
    [string]$Building = "building_001"
)
$ErrorActionPreference = "Stop"
$forge = Split-Path -Parent $PSScriptRoot
$built = Get-ChildItem "$forge\build\$Building\fivem" -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $built) { Write-Host "No resource in build\$Building\fivem: run python -m forge.build first (needs PyMateria for binary files)" -ForegroundColor Red; exit 1 }
$dest = Join-Path $Server "resources\[viewmaxxing]\$($built.Name)"
if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
Copy-Item $built.FullName $dest -Recurse
$cfg = Join-Path $Server "server.cfg"
$line = "ensure $($built.Name)"
if ((Test-Path $cfg) -and -not (Select-String -Path $cfg -SimpleMatch $line -Quiet)) { Add-Content $cfg $line }
Write-Host "Installed $($built.Name) → $dest" -ForegroundColor Green
Write-Host "Restart the server, then FULLY reconnect (stream files are cached; 'restart' alone is not enough)."
Write-Host "In game: F8 console → /$($built.Name.Split('_')[0..1] -join '_') to teleport, /vm_here to pick a new spot."
