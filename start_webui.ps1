[CmdletBinding()]
param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 7860,
    [switch]$Share,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$repositoryRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pipelineRoot = Join-Path $repositoryRoot "badmintondataprocess"
$runtimeSetup = Join-Path $repositoryRoot "setup_runtime.ps1"
$bstRepository = Join-Path $repositoryRoot "third_party\BST-Badminton-Stroke-type-Transformer"
$bstWeights = Join-Path $repositoryRoot "weights\bst\bst_AP_JnB_bone_train_partial_0p25_merged_2.pt"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Install Miniconda, reopen PowerShell, then run this script again."
}

function Resolve-WebUiPython {
    $result = & conda run -n good-badminton python -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $null
    }
    $candidate = ($result | Where-Object { $_ -and $_.Trim() } | Select-Object -Last 1).Trim()
    if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        return $candidate
    }
    return $null
}

$pythonPath = Resolve-WebUiPython
if (-not $pythonPath) {
    throw "Python was not found in the existing good-badminton Conda environment. Run setup_runtime.ps1 after restoring that environment."
}

& $pythonPath -m badminton_data_process.cli verify --profile production --strict --bst-repository $bstRepository --bst-weights $bstWeights 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "The production runtime is incomplete. Repairing the existing good-badminton environment..." -ForegroundColor Cyan
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $runtimeSetup
    if ($LASTEXITCODE -ne 0) {
        throw "The complete BBA runtime could not be installed or verified."
    }
    $pythonPath = Resolve-WebUiPython
}

$arguments = @(
    "-m", "badminton_data_process.cli", "webui",
    "--host", $HostAddress,
    "--port", $Port
)
if ($Share) {
    $arguments += "--share"
}
if ($NoBrowser) {
    $arguments += "--no-browser"
}

Write-Host "Starting BBA WebUI at http://$HostAddress`:$Port" -ForegroundColor Green
Push-Location $pipelineRoot
try {
    & $pythonPath @arguments
}
finally {
    Pop-Location
}
