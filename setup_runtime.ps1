[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$repositoryRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pipelineRoot = Join-Path $repositoryRoot "badmintondataprocess"
$runtimeRequirements = Join-Path $pipelineRoot "requirements-runtime.txt"
$bstRepository = Join-Path $repositoryRoot "third_party\BST-Badminton-Stroke-type-Transformer"
$bstWeights = Join-Path $repositoryRoot "weights\bst\bst_AP_JnB_bone_train_partial_0p25_merged_2.pt"
$bstCommit = "fb9b310bf4c8a8e3d89c75e61bc06a7ac3de62df"
$bstWeightId = "1tcx78bwCO6ZBasw1PHgfyT6BYmzCIqSS"
$bstWeightSha256 = "015F7010526BCC231ECD9006366078943DBD53C0DA8E6D2424B25F0B7A70A502"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open a Conda-enabled PowerShell and run this script again."
}

function Resolve-ProductionPython {
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

$pythonPath = Resolve-ProductionPython
if (-not $pythonPath) {
    throw "The existing Conda environment 'good-badminton' was not found. This script does not create a second environment."
}

Write-Host "Synchronizing the complete BBA runtime in Conda environment: good-badminton" -ForegroundColor Cyan
& $pythonPath -m pip install --disable-pip-version-check --upgrade pip wheel "setuptools<82"
if ($LASTEXITCODE -ne 0) { throw "Failed to update the Python package installer." }

& $pythonPath -m pip install --disable-pip-version-check --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 torchvision==0.26.0+cu128
if ($LASTEXITCODE -ne 0) { throw "Failed to install the CUDA 12.8 PyTorch runtime." }

& $pythonPath -m pip install --disable-pip-version-check -r $runtimeRequirements
if ($LASTEXITCODE -ne 0) { throw "Failed to install the complete BBA runtime requirements." }

# RTMLib declares a dependency on the CPU distribution named 'onnxruntime'.
# BBA intentionally uses onnxruntime-gpu instead; installing RTMLib normally can
# leave both mutually conflicting runtime distributions in the same environment.
& $pythonPath -m pip install --disable-pip-version-check rtmlib==0.0.16 --no-deps
if ($LASTEXITCODE -ne 0) { throw "Failed to install RTMLib with the GPU-safe dependency policy." }

& $pythonPath -m pip install --disable-pip-version-check -e $pipelineRoot
if ($LASTEXITCODE -ne 0) { throw "Failed to install BBA in editable mode." }

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required to install and verify the pinned official BST runtime source."
}
if (-not (Test-Path -LiteralPath $bstRepository -PathType Container)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $bstRepository) | Out-Null
    Write-Host "Installing the pinned official BST source..." -ForegroundColor Cyan
    & git clone --no-checkout https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer.git $bstRepository
    if ($LASTEXITCODE -ne 0) { throw "Failed to clone the official BST source." }
    & git -C $bstRepository checkout --detach $bstCommit
    if ($LASTEXITCODE -ne 0) { throw "Failed to select the verified BST source commit." }
}
$installedBstCommit = (& git -C $bstRepository rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $installedBstCommit -ne $bstCommit) {
    throw "BST source version mismatch. Expected $bstCommit, found $installedBstCommit. Existing third-party files were not overwritten."
}

if (-not (Test-Path -LiteralPath $bstWeights -PathType Leaf)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $bstWeights) | Out-Null
    Write-Host "Downloading the verified official BST integration checkpoint..." -ForegroundColor Cyan
    & $pythonPath -m gdown $bstWeightId --output $bstWeights
    if ($LASTEXITCODE -ne 0) { throw "Failed to download the official BST integration checkpoint." }
}
$installedBstHash = (Get-FileHash -LiteralPath $bstWeights -Algorithm SHA256).Hash
if ($installedBstHash -ne $bstWeightSha256) {
    throw "BST checkpoint checksum mismatch. Expected $bstWeightSha256, found $installedBstHash."
}

$verifyArguments = @(
    "-m", "badminton_data_process.cli", "verify",
    "--profile", "production", "--strict",
    "--bst-repository", $bstRepository,
    "--bst-weights", $bstWeights
)

Write-Host "Running strict production verification..." -ForegroundColor Cyan
& $pythonPath @verifyArguments
if ($LASTEXITCODE -ne 0) {
    throw "The good-badminton environment is installed but did not pass strict production verification."
}

Write-Host "The complete BBA runtime is ready in good-badminton." -ForegroundColor Green
