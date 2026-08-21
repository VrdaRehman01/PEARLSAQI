$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\warda\Downloads\aqi-predictor-v4"
$VenvPython = Join-Path $ProjectRoot ".venv_tf\Scripts\python.exe"

$LogDirectory = Join-Path $ProjectRoot "logs\daily"

New-Item `
    -ItemType Directory `
    -Force `
    -Path $LogDirectory | Out-Null

$Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"

$LogFile = Join-Path `
    $LogDirectory `
    "daily_pipeline_$Timestamp.log"

Set-Location $ProjectRoot

Write-Output "============================================================"
Write-Output "PEARLSAQI AUTOMATED DAILY RUN"
Write-Output "============================================================"
Write-Output "Started : $(Get-Date)"
Write-Output "Project : $ProjectRoot"
Write-Output "Python  : $VenvPython"
Write-Output "Log     : $LogFile"
Write-Output "============================================================"

try {

    if (-not (Test-Path $VenvPython)) {

        throw "Virtual environment Python not found: $VenvPython"
    }

    & $VenvPython `
        -m src.pipelines.daily_forecast_pipeline `
        2>&1 |
        Tee-Object -FilePath $LogFile

    if ($LASTEXITCODE -ne 0) {

        throw "Daily pipeline exited with code $LASTEXITCODE"
    }

    Add-Content `
        -Path $LogFile `
        -Value ""

    Add-Content `
        -Path $LogFile `
        -Value "============================================================"

    Add-Content `
        -Path $LogFile `
        -Value "PIPELINE STATUS: SUCCESS"

    Add-Content `
        -Path $LogFile `
        -Value "Completed : $(Get-Date)"

    Add-Content `
        -Path $LogFile `
        -Value "============================================================"

}
catch {

    Add-Content `
        -Path $LogFile `
        -Value ""

    Add-Content `
        -Path $LogFile `
        -Value "============================================================"

    Add-Content `
        -Path $LogFile `
        -Value "PIPELINE STATUS: FAILED"

    Add-Content `
        -Path $LogFile `
        -Value "Failed : $(Get-Date)"

    Add-Content `
        -Path $LogFile `
        -Value "Error  : $($_.Exception.Message)"

    Add-Content `
        -Path $LogFile `
        -Value "============================================================"

    Write-Error $_

    exit 1
}