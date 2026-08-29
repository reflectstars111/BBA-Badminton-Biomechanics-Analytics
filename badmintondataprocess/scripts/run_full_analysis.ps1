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
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $scriptDirectory "..")).Path
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
    throw "Conda was not found. Install Miniconda and create/update the good-badminton environment from environment.yml."
}

$pythonPath = & conda run -n good-badminton python -c "import sys; print(sys.executable)"
$pythonPath = ($pythonPath | Where-Object { $_ -and $_.Trim() } | Select-Object -Last 1).Trim()
if ($LASTEXITCODE -ne 0 -or -not $pythonPath) {
    throw "Conda environment 'good-badminton' was not found. Run: conda env create -f environment.yml"
}
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python was not found in Conda environment: $pythonPath"
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
