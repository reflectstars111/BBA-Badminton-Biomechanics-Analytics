[CmdletBinding()]
param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 7860,
    [switch]$Share,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pipelineRoot = Join-Path $repositoryRoot "badmintondataprocess"
$environmentFile = Join-Path $pipelineRoot "environment.yml"

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
    Write-Host "Creating the good-badminton Conda environment. This is required only once..." -ForegroundColor Cyan
    Push-Location $pipelineRoot
    try {
        & conda env create -f $environmentFile
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create the good-badminton Conda environment."
        }
    }
    finally {
        Pop-Location
    }
    $pythonPath = Resolve-WebUiPython
}
if (-not $pythonPath) {
    throw "Python was not found in the good-badminton Conda environment."
}

& $pythonPath -c "import gradio, badminton_data_process; major=int(gradio.__version__.split('.')[0]); assert major == 5, f'unsupported Gradio {gradio.__version__}'" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing the WebUI package into the existing environment..." -ForegroundColor Cyan
    Push-Location $pipelineRoot
    try {
        & $pythonPath -m pip install -e ".[ui,yaml]"
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install the WebUI dependencies."
        }
    }
    finally {
        Pop-Location
    }
}

& $pythonPath -c "import rtmlib" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing RTMPose without replacing onnxruntime-gpu..." -ForegroundColor Cyan
    & $pythonPath -m pip install rtmlib==0.0.16 --no-deps
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install RTMPose."
    }
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
