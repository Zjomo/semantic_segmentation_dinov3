param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [Parameter(Mandatory = $false)]
    [string]$OutputCsv = "Results\Temperature_Data\temperature_results.csv",

    [Parameter(Mandatory = $false)]
    [string]$Calibration = "Calibration\raman_temperature_calibration.json",

    [Parameter(Mandatory = $false)]
    [string]$AxisFile = "",

    [Parameter(Mandatory = $false)]
    [string]$PositionsFile = "",

    [Parameter(Mandatory = $false)]
    [string]$Pattern = "*.xls"
)

$scriptPath = Join-Path $PSScriptRoot "raman_temperature_offline.py"

$args = @(
    $scriptPath,
    "--input", $InputPath,
    "--output", $OutputCsv,
    "--calibration", $Calibration,
    "--pattern", $Pattern
)

if ($AxisFile) {
    $args += @("--axis-file", $AxisFile)
}

if ($PositionsFile) {
    $args += @("--positions", $PositionsFile)
}

python @args
