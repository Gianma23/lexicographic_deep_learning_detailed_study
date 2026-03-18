$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

Copy-Item "$PSScriptRoot\graph_pool.py" "$root\internal\graph_pool.py" -Force
Copy-Item "$PSScriptRoot\modules.py" "$root\internal\modules.py" -Force
Copy-Item "$PSScriptRoot\cast_deit_hier.py" "$root\internal\cast_deit_hier.py" -Force
Copy-Item "$PSScriptRoot\factory.py" "$root\factory.py" -Force

Write-Host 'Restored pre-projection files.'
