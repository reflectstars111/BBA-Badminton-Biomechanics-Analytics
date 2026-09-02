[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$InputVideo,

    [string]$RunId = "",

    [string]$Config = "configs/production/full_video_gpu.yaml",

    [string]$RunsDir = "",

    [switch]$ValidateOnly,

    [switch]$Force
)

$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $scriptDirectory "..")).Path
$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $projectRoot "..")).Path
$bstRepository = Join-Path $repositoryRoot "third_party\BST-Badminton-Stroke-type-Transformer"
$bstWeights = Join-Path $repositoryRoot "weights\bst\bst_AP_JnB_bone_train_partial_0p25_merged_2.pt"
$resolvedInput = (Resolve-Path -LiteralPath $InputVideo).Path
$configCandidate = if ([System.IO.Path]::IsPathRooted($Config)) {
    $Config
}
else {
    Join-Path $projectRoot $Config
}
$resolvedConfig = (Resolve-Path -LiteralPath $configCandidate).Path

$condaCommand = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaCommand) {
    throw "Conda was not found. Install Miniconda and restore the good-badminton environment."
}

$pythonPath = & conda run -n good-badminton python -c "import sys; print(sys.executable)"
$pythonPath = ($pythonPath | Where-Object { $_ -and $_.Trim() } | Select-Object -Last 1).Trim()
if ($LASTEXITCODE -ne 0 -or -not $pythonPath) {
    throw "Conda environment 'good-badminton' was not found. The pipeline does not create a second environment."
}
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python was not found in Conda environment: $pythonPath"
}

& $pythonPath -m badminton_data_process.cli verify --profile production --strict --bst-repository $bstRepository --bst-weights $bstWeights
if ($LASTEXITCODE -ne 0) {
    throw "The good-badminton production runtime is incomplete. Run the repository root setup_runtime.ps1 first."
}

$arguments = @(
    "-m", "badminton_data_process.cli",
    "analyze", $resolvedInput,
    "--config", $resolvedConfig
)
if ($RunId) {
    $arguments += @("--run-id", $RunId)
}
if ($RunsDir) {
    $arguments += @("--runs-dir", $RunsDir)
}
if ($ValidateOnly) {
    $arguments += "--preflight-only"
}
if ($Force) {
    $arguments += "--force"
}

Push-Location $projectRoot
try {
    & $pythonPath @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Full analysis failed with exit code $LASTEXITCODE. Inspect the run manifest for the failed stage."
    }
}
finally {
    Pop-Location
}
