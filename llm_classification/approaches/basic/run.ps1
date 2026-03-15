param(
    [switch]$SkipTrain,
    [string]$Message = "auto deploy basic $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
)

$ErrorActionPreference = "Stop"
$root        = Resolve-Path "$PSScriptRoot/../.."
$datasetDir  = "$root/kaggle_dataset/bert-finetuned"
$kaggleId    = "lasseruttert/bert-basic-inference"

if (-not $SkipTrain) {
    Write-Host "=== Training ===" -ForegroundColor Cyan
    conda run --no-capture-output -n kaggle python -u "$PSScriptRoot/train.py"
}

Write-Host "=== Saving model ===" -ForegroundColor Cyan
conda run --no-capture-output -n kaggle python -u "$PSScriptRoot/save.py"
 
Write-Host "=== Uploading dataset ===" -ForegroundColor Cyan
$newDataset = $false
Push-Location $datasetDir
$ErrorActionPreference = "Continue"
conda run -n kaggle kaggle datasets version -p . -m "$Message" 2>&1 | Out-Null
$ErrorActionPreference = "Stop"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Dataset not found, creating..." -ForegroundColor Yellow
    conda run -n kaggle kaggle datasets create -p .
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        Write-Error "Dataset create failed"
    }
    $newDataset = $true
}
Pop-Location

if ($newDataset) {
    Write-Host "=== Waiting 90s for new dataset to be indexed by Kaggle ===" -ForegroundColor Yellow
    Start-Sleep 90
}

Write-Host "=== Pushing kernel ===" -ForegroundColor Cyan
Push-Location $PSScriptRoot
conda run -n kaggle kaggle kernels push -p .
Pop-Location

Write-Host "=== Polling kernel status ===" -ForegroundColor Cyan
do {
    Start-Sleep 30
    $status = conda run -n kaggle kaggle kernels status $kaggleId
    Write-Host $status
} while ($status -notmatch "complete|error|cancel")

if ($status -match "complete") {
    Write-Host "=== Done! ===" -ForegroundColor Green
    Write-Host "  This is a code competition - submission is auto-scored on kernel complete."
    Write-Host "  Check results at: https://www.kaggle.com/competitions/llm-classification-finetuning/submissions"
    Write-Host ""
    Write-Host "  To download kernel output locally:"
    Write-Host ("    Push-Location '" + $PSScriptRoot + "'; conda run -n kaggle kaggle kernels output " + $kaggleId + " -p ./output --force; Pop-Location")
} else {
    Write-Host "=== Kernel ended with status: $status ===" -ForegroundColor Red
    Write-Host "  To fetch output for debugging:"
    Write-Host ("    Push-Location '" + $PSScriptRoot + "'; conda run -n kaggle kaggle kernels output " + $kaggleId + " -p ./output --force; Pop-Location")
}
